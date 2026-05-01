from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from dotenv import load_dotenv
import uuid
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

client = Groq()

engine = create_engine("sqlite:///chat.db")
Session = sessionmaker(bind=engine)
db = Session()

Base = declarative_base()

model         = SentenceTransformer("all-MiniLM-L6-v2")
client_chroma = chromadb.PersistentClient(path="./chroma_data")

collection = client_chroma.get_or_create_collection(
    name="minha_colecao",
    metadata={"hnsw:space": "cosine"}
)

textos = [
    'De segunda-feira à sexta-feira, das 08h às 12h e das 13h30 às 18h.',
    'Aos sábados o atendimento ocorre apenas pela manhã, das 08h às 12h. Não há atendimento aos domingos e feriados.',
    'Os convênios aceitos são: Unimed, plano funerário Nova Esperança, Uniclean e TólioMed.',
    'Também atendemos particular, com pagamento em dinheiro, PIX, cartão de débito ou crédito em até 3x sem juros.',
    'Realizamos exames de sangue no geral, ultrassonografia, exame de raio-x, exames de fezes e urina.',
    'Para exames de sangue é necessário jejum de 8 a 12 horas. Para ultrassonografia abdominal é preciso estar com a bexiga cheia.',
    'Não é permitido fumar dentro do ambiente da clínica nem falar ao celular nas salas de espera.',
    'O paciente deve chegar com ao menos 30 minutos de antecedência e portar documento de identificação com foto e carteirinha do convênio.',
    'A clínica fica localizada na Rua das Acácias, nº 245, bairro Centro, próximo à praça principal.',
    'Para contato e agendamento, ligue para (11) 4002-8922 ou envie mensagem pelo WhatsApp (11) 98765-4321.',
    'As especialidades disponíveis são: clínica geral, cardiologia, pediatria, ginecologia, dermatologia e ortopedia.',
    'O agendamento de consultas pode ser feito por telefone, WhatsApp ou diretamente no site da clínica.',
    'O cancelamento de consultas deve ser feito com no mínimo 24 horas de antecedência, caso contrário é cobrada uma taxa.',
    'Em caso de emergência, a clínica não realiza atendimento de urgência. Procure o pronto-socorro mais próximo ou ligue para o SAMU (192).',
    'A clínica oferece serviço de vacinação para gripe, hepatite B, HPV e febre amarela mediante agendamento prévio.',
    'O ambiente é totalmente acessível para cadeirantes, com rampas, banheiros adaptados e elevador para o segundo andar.'
]
ids = [f'doc{i+1}' for i in range(len(textos))]

vetores = model.encode(textos).tolist()

collection.upsert(
    ids=ids,
    documents=textos,
    embeddings=vetores
)

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    session_id = Column(String)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

def save_message(user_id, session_id, role, content):
    msg = Message(
        user_id=user_id,
        session_id=session_id,
        role=role,
        content=content
    )
    db.add(msg)
    db.commit()


def load_memory(user_id, session_id, limit=10):
    messages = (
        db.query(Message)
        .filter_by(user_id=user_id, session_id=session_id)
        .order_by(Message.id.desc())
        .limit(limit)
        .all()
    )
    
    messages.reverse()
    
    return [
        {"role": m.role, "content": m.content}
        for m in messages
    ]

def consultaChroma(user_input):
    vetor_pergunta = model.encode([user_input]).tolist()

    resultados = collection.query(
        query_embeddings=vetor_pergunta,
        n_results=2
    )

    matches = []

    print(resultados)

    for textos_encontrados in resultados["documents"][0]:
        matches.append(textos_encontrados)

    print(matches)

    return matches

def chat():
    print("Chat com SQLAlchemy + session_id")
    user_id = input("Digite seu user_id: ")
    session_id = str(uuid.uuid4())
    print(f"Sessão iniciada: {session_id}")
    
    while True:
        user_input = input("Você: ")
        if user_input.lower() == "sair":
            break
        
        if user_input.lower() == "/new":
            session_id = str(uuid.uuid4())
            print(f"Nova sessão criada: {session_id}")
            continue
        
        save_message(user_id, session_id, "user", user_input)
        memory = load_memory(user_id, session_id)

        matches = consultaChroma(user_input) 

        messages = [
            {"role": "system", "content": "Você é uma assistente de uma clínica. Utilize isso como contexto: " + str(matches)}
        ] + memory
        
        print(messages)

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )
        
        answer = response.choices[0].message.content
        save_message(user_id, session_id, "assistant", answer)
        print("Bot:", answer)
        print("-" * 50)


if __name__ == "__main__":
    chat()