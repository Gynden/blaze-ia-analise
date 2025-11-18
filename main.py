import os
import io
import base64
import json
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from PIL import Image

# -------------------------------------------------------------------------
# CONFIG BASICA
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY nao encontrada nas variaveis de ambiente!")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BlazeDoubleResponse(BaseModel):
    acao: str          # "ENTRAR_BRANCO" ou "NAO_ENTRAR"
    confianca: float   # 0.0 a 1.0
    justificativa: str


class BlazeCrashResponse(BaseModel):
    acao: str          # "MANTER_ESTRATEGIA_2X" | "CAUTELA" | "EVITAR_ENTRADAS"
    confianca: float
    justificativa: str


# -------------------------------------------------------------------------
# FUNCAO PARA PREPARAR IMAGEM (igual do outro projeto)
# -------------------------------------------------------------------------
def preparar_imagem(upload: UploadFile) -> str:
    """
    Converte a imagem enviada em JPEG comprimido e retorna em base64
    no formato data URL (data:image/jpeg;base64,...).
    """
    img = Image.open(upload.file).convert("RGB")

    max_width = 1280
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# -------------------------------------------------------------------------
# ROTA ROOT (TESTE)
# -------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "API Blaze IA rodando"}


# -------------------------------------------------------------------------
# BLAZE DOUBLE - FOCO NO BRANCO
# -------------------------------------------------------------------------
@app.post("/api/blaze/double", response_model=BlazeDoubleResponse)
async def analisar_blaze_double(image: UploadFile = File(...)):
    if not image or not image.content_type or not image.content_type.startswith("image/"):
        logging.warning("Arquivo invalido recebido em /blaze/double.")
        return BlazeDoubleResponse(
            acao="NAO_ENTRAR",
            confianca=0.0,
            justificativa="Arquivo invalido. Envie um print da tela do Blaze Double."
        )

    try:
        img_b64 = preparar_imagem(image)

        sistema = """
Você é um analista que observa APENAS o print da tela do Blaze Double.
No print aparece o historico recente de resultados em forma de cores (vermelho, preto, branco).

Seu foco é apenas o BRANCO.

Regras importantes:
- Você NÃO consegue prever o futuro com certeza. O jogo é aleatorio e tem vantagem da casa.
- Seu trabalho é descrever o cenario do ponto de vista estatistico VISUAL do historico.
- Leve em conta:
  * Há quantas rodadas aproximadamente nao sai branco.
  * Se a quantidade de brancos parece normal, menor ou maior que o esperado no historico VISIVEL.
  * Se existem sequencias muito longas sem branco no trecho que voce esta vendo.

Objetivo:
- Responder se, olhando apenas o historico visivel no print, faria sentido considerar UMA ENTRADA NO BRANCO AGORA
  ou se e melhor NAO ENTRAR.

Formato de resposta (JSON):
{
  "acao": "ENTRAR_BRANCO" ou "NAO_ENTRAR",
  "confianca": numero entre 0.0 e 1.0,
  "justificativa": "texto curto em portugues explicando o raciocinio, SEM prometer acerto garantido"
}

Se voce tiver duvida, prefira sempre "NAO_ENTRAR".
"""

        usuario_texto = (
            "Analise esse print do Blaze Double focando apenas na possibilidade de entrada no BRANCO. "
            "Considere o historico visivel, sequencias sem branco e frequencia recente. "
            "Se estiver em duvida ou o cenario nao for especial, responda NAO_ENTRAR."
        )

        logging.info(f"[{datetime.utcnow()}] Analise Blaze Double...")

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=300,
            messages=[
                {"role": "system", "content": sistema},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": usuario_texto},
                        {"type": "image_url", "image_url": {"url": img_b64}},
                    ],
                },
            ],
        )

        raw = completion.choices[0].message.content
        logging.info(f"Resposta bruta Double: {raw}")

        data = json.loads(raw)

        acao = str(data.get("acao", "NAO_ENTRAR")).upper().strip()
        if acao not in ("ENTRAR_BRANCO", "NAO_ENTRAR"):
            acao = "NAO_ENTRAR"

        try:
            confianca = float(data.get("confianca", 0.0))
        except Exception:
            confianca = 0.0

        justificativa = str(data.get("justificativa", "")).strip()
        if not justificativa:
            justificativa = "Cenario indefinido. Melhor nao entrar no branco."

        return BlazeDoubleResponse(
            acao=acao,
            confianca=confianca,
            justificativa=justificativa
        )

    except Exception:
        logging.exception("Erro ao analisar Blaze Double")
        return BlazeDoubleResponse(
            acao="NAO_ENTRAR",
            confianca=0.0,
            justificativa="Erro ao analisar o print do Blaze Double. Tente novamente mais tarde."
        )


# -------------------------------------------------------------------------
# BLAZE CRASH - FOCO NA ESTRATEGIA 2X
# -------------------------------------------------------------------------
@app.post("/api/blaze/crash", response_model=BlazeCrashResponse)
async def analisar_blaze_crash(image: UploadFile = File(...)):
    if not image or not image.content_type or not image.content_type.startswith("image/"):
        logging.warning("Arquivo invalido recebido em /blaze/crash.")
        return BlazeCrashResponse(
            acao="CAUTELA",
            confianca=0.0,
            justificativa="Arquivo invalido. Envie um print da tela do Blaze Crash."
        )

    try:
        img_b64 = preparar_imagem(image)

        sistema = """
Você é um analista que observa APENAS o print da tela do Blaze Crash.
No print aparece o historico recente de multiplicadores (por exemplo 1.20x, 3.50x, 10x, etc).

Seu foco é avaliar a viabilidade da estrategia de sempre sair em 2x,
com base APENAS no historico visivel, sem prometer acerto.

Regras importantes:
- Voce NAO consegue prever exatamente onde o proximo crash vai parar.
- O jogo e aleatorio e tem vantagem da casa. Seu papel e descrever risco e comportamento recente.
- Leve em conta:
  * Qual percentual aproximado de rodadas, no historico visivel, teria batido 2x ou mais.
  * Quantas rodadas recentes pararam abaixo de 2x.
  * Se ha uma sequencia longa de crashes baixos (1.xx) ou se existe boa variacao.

Objetivo:
- Classificar o cenario para a estrategia 2x em tres possiveis saidas:
  * "MANTER_ESTRATEGIA_2X": historico visivel mostra porcentagem razoavel de rodadas acima de 2x.
  * "CAUTELA": ha sinal misto, com alguns periodos ruins, exigindo muita disciplina de gerenciamento.
  * "EVITAR_ENTRADAS": historico recente e dominado por crashes baixos, com poucas rodadas acima de 2x.

Formato de resposta (JSON):
{
  "acao": "MANTER_ESTRATEGIA_2X" ou "CAUTELA" ou "EVITAR_ENTRADAS",
  "confianca": numero entre 0.0 e 1.0,
  "justificativa": "texto curto em portugues explicando o raciocinio, SEM prometer lucro garantido"
}

Se estiver em duvida, penda para "CAUTELA" ou "EVITAR_ENTRADAS".
"""

        usuario_texto = (
            "Analise esse print do Blaze Crash pensando na estrategia de sempre sair em 2x. "
            "Olhe o historico de multiplicadores visivel e classifique o cenario conforme seu manual."
        )

        logging.info(f"[{datetime.utcnow()}] Analise Blaze Crash...")

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=300,
            messages=[
                {"role": "system", "content": sistema},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": usuario_texto},
                        {"type": "image_url", "image_url": {"url": img_b64}},
                    ],
                },
            ],
        )

        raw = completion.choices[0].message.content
        logging.info(f"Resposta bruta Crash: {raw}")

        data = json.loads(raw)

        acao = str(data.get("acao", "CAUTELA")).upper().strip()
        if acao not in ("MANTER_ESTRATEGIA_2X", "CAUTELA", "EVITAR_ENTRADAS"):
            acao = "CAUTELA"

        try:
            confianca = float(data.get("confianca", 0.0))
        except Exception:
            confianca = 0.0

        justificativa = str(data.get("justificativa", "")).strip()
        if not justificativa:
            justificativa = "Cenario misto. Uso da estrategia 2x exige cautela."

        return BlazeCrashResponse(
            acao=acao,
            confianca=confianca,
            justificativa=justificativa
        )

    except Exception:
        logging.exception("Erro ao analisar Blaze Crash")
        return BlazeCrashResponse(
            acao="CAUTELA",
            confianca=0.0,
            justificativa="Erro ao analisar o print do Blaze Crash. Tente novamente mais tarde."
        )
