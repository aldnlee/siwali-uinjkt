from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_7PWzvf_CHQfZqkisrFkKkSCBReR12XiMpxvpe5XLtj3sRxWhxa8vVErafHmRg5iXrad7Mk")
index = pc.Index("uin-jkt-index")

# Tarik 5 sampel data untuk melihat format ID-nya
sample = index.query(
    vector=[0]*384, # Sesuaikan dengan dimensi model Anda
    top_k=5,
    include_metadata=False
)

for match in sample['matches']:
    print(f"ID ditemukan: {match['id']}")