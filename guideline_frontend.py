import streamlit as st

# SQLite3の互換性対応（他のimportの前に配置）
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from guideline_management_system import GuidelineManagementSystem
import os

# Initialize the GuidelineManagementSystem
@st.cache_resource
def init_gms():
    # Streamlit Cloudの環境変数を設定
    for key in st.secrets:
        os.environ[key] = st.secrets[key]
    
    # データベースとベクトルストアのパスを相対パスに変更
    sqlite_path = "data/guidelines.db"
    vector_store_path = "data/vector_store"
    
    # ディレクトリが存在しない場合は作成
    os.makedirs("data", exist_ok=True)
    
    return GuidelineManagementSystem(sqlite_path, vector_store_path)

def main():
    st.title("Guideline Query System")
    
    # Initialize system
    gms = init_gms()
    
    # Main query interface
    st.header("Ask Questions About Guidelines")
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("Submit Question"):
        if query:
            try:
                with st.spinner("Searching guidelines..."):
                    response, guidelines, relevant_chunks = gms.answer_query_with_chunks(query)
                
                # Display response
                st.subheader("Answer")
                st.write(response)
                
                # Display referenced guidelines
                st.subheader("Referenced Guidelines")
                for g in guidelines:
                    with st.expander(f"{g['title']} (ID: {g['id']})"):
                        st.write(f"Version: {g['version']}")
                        st.write(f"Adopted: {g['adopted_date']}")
                        if 'summary' in g and g['summary']:
                            st.write("Summary:", g['summary'])
                
                # Display relevant chunks used for the answer
                st.subheader("Reference Passages")
                
                # Create a mapping of guideline IDs to titles
                guideline_titles = {g['id']: g['title'] for g in guidelines}
                
                for i, chunk in enumerate(relevant_chunks, 1):
                    guideline_id = chunk.metadata['guideline_id']
                    guideline_title = guideline_titles.get(guideline_id, "Unknown Guideline")
                    with st.expander(f"Reference {i} from {guideline_title} (ID: {guideline_id})"):
                        st.write(chunk.page_content)
            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()