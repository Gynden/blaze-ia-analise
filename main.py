from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os

# Se quiser usar a Groq depois, já deixei import pronto:
# from groq import Groq

app = FastAPI()

# Domínios permitidos (GitHub Pages + local)
origins = [
    "https://gynden.github.io",
    "https://gynden.github.io/",
    "http://localhost",
    "http://127.0.0.1:5500",
    "http://localhost:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Se for usar Groq depois, é só descomentar
# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class AnaliseResponse(BaseModel):
    acao: str
    confianca: float
    justificativa: str


@app.get("/")
async def root():
    return {"status": "online", "message": "Blaze IA OK"}


@app.post("/api/analisar", response_model=AnaliseResponse)
async def analisar_imagem(
    image: UploadFile = File(...),
    modo: str = Form("double"),  # "double" ou "crash"
):
    """
    Endpoint que o frontend chama com:
      - image: arquivo de imagem (print da Blaze)
      - modo: "double" (branco) ou "crash" (2x)
    """

    conteudo = await image.read()

    # Se depois você quiser mandar a imagem em base64 pra Groq, já está pronto:
    img_b64 = base64.b64encode(conteudo).decode("utf-8")

    contexto = (
        "Blaze Double focado em pegar apenas o branco."
        if modo == "double"
        else "Blaze Crash focado em entradas até 2x."
    )

    # 🔴 Por enquanto: resposta fake só pra garantir que tudo está conectado.
    # Depois trocamos por uma chamada real pra Groq (tipo no outro projeto).
    return AnaliseResponse(
        acao="NAO_OPERAR",
        confianca=0.0,
        justificativa=(
            f"Backend da Blaze IA está online. "
            f"Modo: {modo}. Lógica de análise será configurada em seguida."
        ),
    )
