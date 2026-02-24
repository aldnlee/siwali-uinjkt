from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_6srW5C_4BwsbzHmD9zXpUb2p66RxTVvgPdNPAQhK5Y2U4jdofUCQrmKm7V8Be4cTtUPca3")
index = pc.Index("uin-jkt-index")

# Tarik 5 sampel data untuk melihat format ID-nya
sample = index.query(
    vector=[0]*384, # Sesuaikan dengan dimensi model Anda
    top_k=5,
    include_metadata=False
)

for match in sample['matches']:
    print(f"ID ditemukan: {match['id']}")