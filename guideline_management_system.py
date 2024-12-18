import os
import time
import openai
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sqlite3
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_extraction_chain, LLMChain
from langchain_openai import ChatOpenAI
from langsmith import wrappers, traceable, Client
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.runnable import RunnablePassthrough



class GuidelineManagementSystem:
    def __init__(self, sqlite_path: str, vector_store_path: str):
        
        self.failed_files = []  # 失敗したファイルを記録するリスト
        """
        Initialize the Guideline Management System
        
        Args:
            sqlite_path: Path to SQLite database
            vector_store_path: Path to vector store
        """
        # Load environment variables from .env file
        load_dotenv()
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = wrappers.wrap_openai(openai.Client())

        # Validate required environment variables
        required_env_vars = [
            "OPENAI_API_KEY", 
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_ENDPOINT",
            "LANGCHAIN_PROJECT"
        ]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

        self.sqlite_path = sqlite_path
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        self.langsmith_client = Client()
        
        # Initialize databases
        self._init_sqlite()
        self._init_vector_store()

    def _init_sqlite(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS guidelines (
                    id VARCHAR(20) PRIMARY KEY,
                    title TEXT NOT NULL,
                    version VARCHAR(10),
                    adopted_date DATE,
                    pdf_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS guideline_summaries (
                    guideline_id VARCHAR(20) REFERENCES guidelines(id),
                    summary TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (guideline_id)
                )
            """)
            
            conn.commit()

    def _init_vector_store(self):
        """Initialize Chroma vector store"""
        import chromadb
        
        # 新しい推奨される方法でChromaDBクライアントを初期化
        client = chromadb.PersistentClient(
            path=self.vector_store_path
        )
        
        # クライアント設定を含めてChromaを初期化
        self.vector_store = Chroma(
            collection_name="guidelines_collection",
            embedding_function=self.embeddings,
            client=client
        )
        
        @traceable
        def _extract_metadata(self, document) -> Dict:
            """Extract metadata from document text using LangChain"""
            # 最初のページのテキストを使用して基本メタデータを抽出
            first_page_text = document[0].page_content
            
            # 最初の5ページ（または利用可能な全ページ）のトを結合してサマリー用に使用
            summary_text = ""
            for i in range(min(5, len(document))):
                summary_text += document[i].page_content + "\n"

            # 基本メタデータ抽出のスキーマ
            basic_schema = {
                "title": "GuidelineMetadata",
                "description": "Basic metadata extraction schema for guidelines",
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title of the guideline"},
                    "version": {"type": "string", "description": "Version number of the guideline"},
                    "adopted_date": {"type": "string", "description": "Date when the guideline was adopted"}
                },
                "required": ["title", "version", "adopted_date"]
            }

            # サマリー抽出用のスキーマ
            summary_schema = {
                "title": "GuidelineSummary",
                "description": "Summary extraction schema for guidelines",
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Comprehensive summary of the guideline"}
                },
                "required": ["summary"]
            }

            try:
                # 本デの出
                basic_extraction_model = self.llm.with_structured_output(basic_schema)
                basic_metadata = basic_extraction_model.invoke(
                    f"""Please extract the following information from this text:
                    {first_page_text[:4000]}
                    
                    Please extract:
                    - The title of the guideline
                    - The version number
                    - The adoption date
                    
                    If you can't find exact information, please provide the best estimate based on available content.
                    """
                )

                # サマリーの抽出
                summary_extraction_model = self.llm.with_structured_output(summary_schema)
                summary_result = summary_extraction_model.invoke(
                    f"""Please provide a comprehensive summary of this guideline based on the following text:
                    {summary_text[:8000]}
                    
                    Focus on the main objectives, key recommendations, and important points.
                    """
                )

                # 結果の結合
                result = {
                    "title": basic_metadata.get("title"),
                    "version": basic_metadata.get("version"),
                    "adopted_date": basic_metadata.get("adopted_date"),
                    "summary": summary_result.get("summary")
                }

                print("Extraction result:", result)  # デバッグ用
                return result

            except Exception as e:
                print(f"Extraction error: {str(e)}")  # デバッグ用
                return {}
        
      
        def process_guideline(self, pdf_path: str) -> str:
            """
            Process a single guideline PDF
            
            Args:
                pdf_path: Path to the PDF file
                
            Returns:
                Guideline ID of the processed document or existing ID if already processed
            """
            # ��存のPDFをチェック
            with sqlite3.connect(self.sqlite_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT id FROM guidelines WHERE pdf_path = ?", (pdf_path,))
                existing = cur.fetchone()
                
                if existing:
                    print(f"PDF already processed: {pdf_path} (ID: {existing[0]})")
                    return existing[0]

            try:
                # Load and split document
                loader = PDFPlumberLoader(pdf_path)
                document = loader.load()
                
                print(f"Processing new PDF: {pdf_path}")
                print(f"Number of pages: {len(document)}")
                
                # テキスト抽出の認
                if document and document[0].page_content:
                    print("First page content available")
                else:
                    print("No content extracted from PDF")
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                splits = splitter.split_documents(document)
                
                # Extract metadata
                metadata = self._extract_metadata(splits)
                
                # Generate unique ID
                guideline_id = f"GL{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # メタデータにguideline_idを追加
                for split in splits:
                    split.metadata['guideline_id'] = guideline_id
                
                # Store in vector store
                self.vector_store.add_documents(
                    documents=splits,
                    ids=[f"{guideline_id}-{i}" for i in range(len(splits))]
                )
                
                # リトライロジックを実装したSQLite操作
                max_retries = 3
                retry_delay = 2

                for attempt in range(max_retries):
                    try:
                        with sqlite3.connect(self.sqlite_path, timeout=30) as conn:
                            cur = conn.cursor()
                            
                            # Insert guideline metadata
                            cur.execute("""
                                INSERT INTO guidelines (id, title, version, adopted_date, pdf_path)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                guideline_id,
                                metadata.get('title'),
                                metadata.get('version'),
                                metadata.get('adopted_date'),
                                pdf_path
                            ))
                            
                            # Insert summary
                            if metadata.get('summary'):
                                cur.execute("""
                                    INSERT INTO guideline_summaries (guideline_id, summary)
                                    VALUES (?, ?)
                                """, (guideline_id, metadata.get('summary')))
                            
                            conn.commit()
                            break  # 成功時にループを抜ける
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and attempt < max_retries - 1:
                            print(f"Database is locked, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                        else:
                            raise e
                
                return guideline_id
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                return ""

      
        def search_guidelines(self, query: str, limit: int = 5) -> List[Dict]:
            """
            Search guidelines using semantic search
            
            Args:
                query: Search query
                limit: Maximum number of results
                
            Returns:
                List of matching guidelines with metadata
            """
            # Perform vector search
            docs = self.vector_store.similarity_search(query, k=limit)
            
            # Get metadata from SQLite
            guideline_ids = list(set(doc.metadata['guideline_id'] for doc in docs))
            
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                
                results = []
                for guideline_id in guideline_ids:
                    cur.execute("""
                        SELECT g.*, s.summary
                        FROM guidelines g
                        LEFT JOIN guideline_summaries s ON g.id = s.guideline_id
                        WHERE g.id = ?
                    """, (guideline_id,))
                    
                    result = dict(cur.fetchone())
                    results.append(result)
            
            return results

     
        def get_similar_guidelines(self, guideline_id: str, limit: int = 3) -> List[Dict]:
            """
            Find similar guidelines to a given guideline
            
            Args:
                guideline_id: ID of the reference guideline
                limit: Maximum number of results
                
            Returns:
                List of similar guidelines
            """
            with sqlite3.connect(self.sqlite_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT title FROM guidelines WHERE id = ?", (guideline_id,))
                title = cur.fetchone()[0]
            
            return self.search_guidelines(title, limit)

       
        def process_directory(self, directory_path: str) -> List[str]:
            """
            Process all PDF files in a directory and its subdirectories
            
            Args:
                directory_path: Path to the directory containing PDF files
                
            Returns:
                List of processed guideline IDs
            """
            processed_ids = []
            skipped_files = []  # 既に処理済みのファイルを記録
            
            # Walk through directory and subdirectories
            for root, dirs, files in os.walk(directory_path):
                # Process each PDF file
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        try:
                            guideline_id = self.process_guideline(pdf_path)
                            if guideline_id:
                                if pdf_path not in [f['path'] for f in self.failed_files]:
                                    # 新規処理の場合
                                    processed_ids.append(guideline_id)
                                    print(f"Successfully processed: {pdf_path}")
                                else:
                                    # 既存のファイルの場合
                                    skipped_files.append(pdf_path)
                                    print(f"Skipped (already exists): {pdf_path}")
                            else:
                                self.failed_files.append({
                                    'path': pdf_path,
                                    'reason': 'Processing failed',
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                                print(f"Failed to process: {pdf_path}")
                        except Exception as e:
                            self.failed_files.append({
                                'path': pdf_path,
                                'reason': str(e),
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            print(f"Error processing {pdf_path}: {str(e)}")
            
            # 処理結果のサマリーを表示
            print("\n=== Processing Summary ===")
            print(f"Total files found: {len(processed_ids) + len(skipped_files) + len(self.failed_files)}")
            print(f"Newly processed: {len(processed_ids)}")
            print(f"Skipped (already exists): {len(skipped_files)}")
            print(f"Failed to process: {len(self.failed_files)}")
            
            # 失敗したファイルの詳細を表示
            if self.failed_files:
                print("\n=== Failed Files Details ===")
                for failed in self.failed_files:
                    print(f"\nFile: {failed['path']}")
                    print(f"Time: {failed['timestamp']}")
                    print(f"Reason: {failed['reason']}")
            
            return processed_ids

        def _get_relevant_guidelines(self, query: str, limit: int = 3) -> List[Dict]:
            # 1. まずVector Storeで意味的な検索を実行
            docs = self.vector_store.similarity_search(query, k=limit)
            
            # 2. 検索結果から関連するガイドラインIDを抽出
            guideline_ids = list(set(doc.metadata['guideline_id'] for doc in docs))
            
            # 3. SQLiteから該当するガイドラインの詳細情報を取得
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                
                results = []
                for guideline_id in guideline_ids:
                    cur.execute("""
                        SELECT g.id, g.title, g.version, g.adopted_date, s.summary
                        FROM guidelines g
                        LEFT JOIN guideline_summaries s ON g.id = s.guideline_id
                        WHERE g.id = ?
                    """, (guideline_id,))
                    
                    row = cur.fetchone()
                    if row:
                        results.append(dict(row))
            
            return results

        def _get_relevant_chunks(self, query: str, guideline_ids: List[str], chunks_per_guideline: int = 3) -> List[Dict]:
            """
            Get relevant text chunks from specified guidelines
            
            Args:
                query: User query
                guideline_ids: List of guideline IDs to search within
                chunks_per_guideline: Maximum chunks to retrieve per guideline
                
            Returns:
                List of relevant text chunks with metadata
            """
            # Create compressor for contextual chunk retrieval
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": chunks_per_guideline * len(guideline_ids)}
                ),
                base_compressor=compressor
            )
            
            # Filter for specified guideline IDs and retrieve relevant chunks
            docs = retriever.invoke(query)
            filtered_docs = [
                doc for doc in docs 
                if any(doc.metadata['guideline_id'].startswith(gid) for gid in guideline_ids)
            ]
            
            return filtered_docs

        def answer_query_with_chunks(self, query: str, max_guidelines: int = 3) -> Tuple[str, List[Dict], List[Dict]]:
            """
            2段階の検索プロセスを実行し、回答、ガイドライン情報、使用したチャンクを返す
            
            Args:
                query: ユーザーからの質問
                max_guidelines: 検索する最大ガイドライン数
                
            Returns:
                Tuple containing:
                - Generated response (str)
                - List of relevant guidelines metadata (List[Dict])
                - List of relevant text chunks used for the answer (List[Dict])
            """
            # 1. サマリーベースでの関連ガイドライン特定
            relevant_guidelines = self._get_relevant_guidelines(query, limit=max_guidelines)
            
            # デバッグ出力
            print("\nRetrieved guidelines metadata:")
            for g in relevant_guidelines:
                print(f"ID: {g.get('id')}")
                print(f"Title: {g.get('title')}")
                print(f"Version: {g.get('version')}")
                print(f"Adopted: {g.get('adopted_date')}")
                print("---")
            
            guideline_ids = [g['id'] for g in relevant_guidelines]
            
            # 2. 特定されたガイドラインからの詳細な情報抽出
            relevant_chunks = self._get_comprehensive_chunks(
                query=query,
                guideline_ids=guideline_ids,
                min_chunks_per_guideline=3,
                similarity_threshold=0.6
            )
            
            # 3. レスポンス生成
            response = self._generate_response(query, relevant_chunks, relevant_guidelines)
            
            return response, relevant_guidelines, relevant_chunks

        def _generate_response(self, query: str, relevant_chunks: List[Dict], guidelines_metadata: List[Dict]) -> str:
            """
            Generate response using LLM based on relevant chunks and metadata
            
            Args:
                query: User query
                relevant_chunks: List of relevant text chunks
                guidelines_metadata: List of metadata for relevant guidelines
                
            Returns:
                Generated response
            """
            # Format context and guidelines information
            context_text = "\n\n".join([
                f"From {chunk.metadata['guideline_id']} (Page {chunk.metadata.get('page_number', 'N/A')}):\n{chunk.page_content}" 
                for chunk in relevant_chunks
            ])
            
            guidelines_text = "\n".join([
                f"- {g['title']} (ID: {g['id']}, Version: {g['version']}, Adopted: {g['adopted_date']})"
                for g in guidelines_metadata
            ])
            
            # Create prompt template
            prompt = PromptTemplate.from_template(
                """You are an expert in analyzing and explaining guidelines.
                
                Question: {query}
                
                Relevant guidelines:
                {guidelines}
                
                Context from guidelines:
                {context}
                
                Please provide a comprehensive answer that:
                1. Directly addresses the question
                2. Cites specific guidelines and page numbers when referencing information
                3. Provides clear explanations and examples where appropriate
                4. Highlights any important caveats or considerations
                
                Answer:"""
            )
            
            # Create pipeline using the new syntax
            chain = (
                {"query": RunnablePassthrough(), 
                 "context": lambda _: context_text, 
                 "guidelines": lambda _: guidelines_text}
                | prompt 
                | ChatOpenAI(temperature=0)
            )
            
            # Run the chain
            response = chain.invoke(query)
            
            return response.content

        def _get_comprehensive_chunks(
            self, 
            query: str, 
            guideline_ids: List[str],
            min_chunks_per_guideline: int = 3,
            similarity_threshold: float = 0.6
        ) -> List[Dict]:
            """
            Get comprehensive chunks from guidelines with improved metadata
            
            Args:
                query: Search query
                guideline_ids: List of guideline IDs to search within
                min_chunks_per_guideline: Minimum chunks to retrieve per guideline
                similarity_threshold: Similarity threshold for relevance
                
            Returns:
                List of relevant chunks with enhanced metadata
            """
            print(f"\n=== Debug: Chunk Extraction Start ===")
            print(f"Query: {query}")
            print(f"Target guideline IDs: {guideline_ids}")
            
            base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": min_chunks_per_guideline * len(guideline_ids) * 2}
            )

            compressed_chunks = []
            for guideline_id in guideline_ids:
                try:
                    chunks = base_retriever.invoke(query)
                    filtered_chunks = [
                        chunk for chunk in chunks 
                        if chunk.metadata.get('guideline_id') == guideline_id
                    ]
                    
                    # Sort chunks by relevance and take top N
                    if filtered_chunks:
                        # Ensure each chunk has complete metadata
                        for chunk in filtered_chunks[:min_chunks_per_guideline]:
                            if 'page_number' not in chunk.metadata:
                                chunk.metadata['page_number'] = 'N/A'
                        compressed_chunks.extend(filtered_chunks[:min_chunks_per_guideline])
                    
                except Exception as e:
                    print(f"Error processing guideline {guideline_id}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())

            print(f"\nTotal final chunks: {len(compressed_chunks)}")
            return compressed_chunks

        def interactive_query(self):
            """
            Interactive query interface for the system
            """
            print("Welcome to the Guideline Query System!")
            print("Available commands:")
            print("- 'inspect': Show vector store status")
            print("- 'exit': Quit the system")
            print("\nEnter your query or command:")
            
            while True:
                query = input("\nYour query or command: ").strip()
                
                if query.lower() == 'exit':
                    break
                elif query.lower() == 'inspect':
                    print("\n=== Vector Store Inspection Results ===")
                    doc_count = self.inspect_vector_store()
                    print(f"\nTotal documents in vector store: {doc_count}")
                    continue
                
                try:
                    response, guidelines, relevant_chunks = self.answer_query_with_chunks(query)
                    
                    print("\n=== Response ===")
                    print(response)
                    
                    print("\n=== Referenced Guidelines ===")
                    for g in guidelines:
                        print(f"- {g['title']} (Version: {g['version']}, Adopted: {g['adopted_date']})")
                        
                except Exception as e:
                    print(f"\nError processing query: {str(e)}")

        def inspect_vector_store(self):
            """
            ベクトルストアの内容を確認するデバッグメソッド
            """
            print("\n=== Vector Store Inspection ===")
            
            try:
                # コレクションの基本情報を取得
                collection = self.vector_store._collection
                
                # 保存されているドキュメント数を確認
                count = collection.count()
                print(f"Total documents in store: {count}")
                
                if count > 0:
                    # サンプルのドキュメントを取得
                    results = collection.get()
                    print(f"\nMetadata sample (up to 3 documents):")
                    for i, (id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
                        if i >= 3:  # 最初の3件のみ表示
                            break
                        print(f"\nDocument {i+1}:")
                        print(f"ID: {id}")
                        print(f"Metadata: {metadata}")
                        print(f"Content preview: {results['documents'][i][:200]}...")
                
                return count
                
            except Exception as e:
                print(f"Error inspecting vector store: {str(e)}")
                return 0

if __name__ == "__main__":
    # テストやデバッグ用のコードをここに書く（必要な場合）
    pass

