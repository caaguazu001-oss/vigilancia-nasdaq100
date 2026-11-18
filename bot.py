#!/usr/bin/env python3
"""
Bot de vigilancia del Nasdaq-100 (velas de 1 día).

Envía una alerta cuando una acción:
  1) cae por debajo de RSI(14) = 35
  2) cae por debajo de su MA50  (media móvil simple de 50 días)
  3) cae por debajo de su MA200 (media móvil simple de 200 días)
Basta con que ocurra una de las tres. Si no ocurre ninguna, no envía nada.

"Cae por debajo" = cruce: en la vela anterior estaba en o por encima del nivel
y en la vela actual está por debajo (MODO_ALERTA=estado cambia esto, ver README).

Revisa dos veces por día hábil de la bolsa de Nueva York:
  - apertura: ~15 min después de abrir (vela del día en curso, señal provisional)
  - cierre:   ~15 min después de cerrar (vela del día cerrada, señal confirmada)
Cada señal se avisa una sola vez por acción y por día.

Uso:
  python bot.py                     automático (decide según la hora de Nueva York)
  python bot.py --ventana prueba    mensaje de prueba con el estado actual
  python bot.py --ventana cierre    fuerza una revisión ahora
  python bot.py --seco ...          no envía ni guarda nada; imprime el mensaje
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import smtplib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def _cargar_dotenv(ruta: Path) -> None:
    """Carga un archivo .env sencillo (solo para uso local; en GitHub se usan Secrets)."""
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


_cargar_dotenv(BASE / ".env")


def _env(nombre: str, defecto: str = "") -> str:
    valor = os.getenv(nombre)
    return valor.strip() if valor and valor.strip() else defecto


# ---------------------------------------------------------------- configuración
NY = ZoneInfo("America/New_York")
TZ_LOCAL = ZoneInfo(_env("ZONA_HORARIA_LOCAL", "America/Asuncion"))
NOMBRE_TZ_LOCAL = _env("NOMBRE_ZONA_LOCAL", "Asunción")

ARCHIVO_ESTADO = Path(_env("ARCHIVO_ESTADO", str(BASE / "state.json")))
ARCHIVO_TICKERS = Path(_env("ARCHIVO_TICKERS", str(BASE / "tickers.txt")))

RSI_PERIODO = 14
RSI_UMBRAL = float(_env("RSI_UMBRAL", "35"))
MA_RAPIDA = 50
MA_LENTA = 200
MODO = _env("MODO_ALERTA", "cruce").lower()          # "cruce" o "estado"

# Ventanas en hora de Nueva York (el horario de verano lo resuelve zoneinfo).
HORA_APERTURA = dtime(9, 40)    # la bolsa abre 9:30
FIN_APERTURA = dtime(12, 0)
HORA_CIERRE = dtime(16, 10)     # la bolsa cierra 16:00
FIN_CIERRE = dtime(23, 30)
MAX_INTENTOS = 2                # fallos seguidos antes de avisar y rendirse esa ventana
DIAS_HISTORIAL = 15             # días que se guardan las alertas enviadas

ETIQUETAS = {"rsi": f"RSI(14) < {RSI_UMBRAL:g}", "ma50": "Bajo MA50", "ma200": "Bajo MA200"}
DIAS_SEMANA = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

log = logging.getLogger("vigilancia")


class DatosIncompletos(RuntimeError):
    pass


# ------------------------------------------------------------------ utilidades
def num(x: float, dec: int = 2) -> str:
    """Número con formato español: 1.234,56"""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x: float, dec: int = 1) -> str:
    signo = "+" if x > 0 else ("−" if x < 0 else "")
    return f"{signo}{num(abs(x), dec)} %"


def es_dia_habil(d: date) -> bool:
    """Lunes a viernes que no sean feriado de la bolsa de NY (Nasdaq usa los mismos)."""
    if d.weekday() >= 5:
        return False
    try:
        import holidays
        return d not in holidays.financial_holidays("NYSE", years=d.year)
    except Exception as e:  # si falla la librería, los datos de Yahoo hacen de red de seguridad
        log.warning("No pude consultar feriados (%s); asumo día hábil", e)
        return True


def hora_local(d: date, t: dtime) -> str:
    return datetime.combine(d, t, tzinfo=NY).astimezone(TZ_LOCAL).strftime("%H:%M")


def leer_tickers(ruta: Path = None) -> list[str]:
    ruta = ruta or ARCHIVO_TICKERS
    vistos, lista = set(), []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        t = linea.split("#", 1)[0].strip().upper()
        if t and t not in vistos:
            vistos.add(t)
            lista.append(t)
    if not lista:
        raise RuntimeError(f"{ruta.name} no contiene acciones")
    return lista


# ---------------------------------------------------------------------- estado
def cargar_estado() -> dict:
    try:
        return json.loads(ARCHIVO_ESTADO.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        log.warning("state.json dañado; empiezo con estado vacío")
        return {}


def guardar_estado(estado: dict, hoy: date) -> None:
    limite = (hoy - timedelta(days=DIAS_HISTORIAL)).isoformat()
    estado["enviadas"] = {k: v for k, v in estado.get("enviadas", {}).items()
                          if k.rsplit("|", 1)[-1] >= limite}
    estado["ventanas"] = {k: v for k, v in estado.get("ventanas", {}).items() if k >= limite}
    ARCHIVO_ESTADO.write_text(json.dumps(estado, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                              encoding="utf-8")


def decidir_ventana(ahora_ny: datetime, estado: dict) -> str | None:
    """Devuelve 'apertura', 'cierre' o None si ahora no toca revisar."""
    hoy = ahora_ny.date()
    if not es_dia_habil(hoy):
        return None
    t = ahora_ny.time()
    registro = estado.get("ventanas", {}).get(hoy.isoformat(), {})
    for nombre, ini, fin in (("apertura", HORA_APERTURA, FIN_APERTURA),
                             ("cierre", HORA_CIERRE, FIN_CIERRE)):
        if ini <= t < fin and registro.get(nombre, {}).get("estado") not in ("ok", "agotada"):
            return nombre
    return None


# ------------------------------------------------------------------- datos
def descargar_cierres(tickers: list[str]) -> pd.DataFrame:
    """Cierres diarios (ajustados por splits, no por dividendos, igual que TradingView)."""
    import yfinance as yf

    def bajar(lista: list[str]) -> pd.DataFrame:
        yahoo = [t.replace(".", "-") for t in lista]
        for intento in range(3):
            try:
                df = yf.download(yahoo, period="2y", interval="1d", auto_adjust=False,
                                 actions=False, progress=False, threads=True, timeout=30)
                if df is None or df.empty:
                    raise RuntimeError("Yahoo devolvió una tabla vacía")
                cierres = df["Close"]
                if isinstance(cierres, pd.Series):
                    cierres = cierres.to_frame(yahoo[0])
                return cierres.rename(columns=dict(zip(yahoo, lista)))
            except Exception as e:
                log.warning("Descarga fallida (intento %d/3): %s", intento + 1, e)
                time.sleep(10 * (intento + 1))
        raise RuntimeError("No pude descargar datos de Yahoo Finance tras 3 intentos")

    cierres = bajar(tickers)
    faltan = [t for t in tickers if t not in cierres.columns or cierres[t].dropna().empty]
    if faltan and len(faltan) < len(tickers):
        log.info("Reintentando %d acciones sin datos: %s", len(faltan), ", ".join(faltan))
        time.sleep(5)
        try:
            extra = bajar(faltan)
            for t in extra.columns:
                if not extra[t].dropna().empty:
                    cierres[t] = extra[t]
        except RuntimeError as e:
            log.warning("%s", e)
    indice = pd.to_datetime(cierres.index)
    if indice.tz is not None:
        indice = indice.tz_localize(None)
    cierres.index = indice.normalize()
    return cierres.sort_index()


# ------------------------------------------------------------- indicadores
def calcular_rsi(cierre: pd.Series, periodo: int = RSI_PERIODO) -> pd.Series:
    """RSI de Wilder (el mismo que TradingView): semilla = media simple, luego suavizado RMA."""
    c = cierre.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= periodo:
        return pd.Series(out, index=cierre.index)
    d = np.diff(c)
    g, p = np.clip(d, 0, None), np.clip(-d, 0, None)
    media_g, media_p = g[:periodo].mean(), p[:periodo].mean()

    def valor(mg, mp):
        return 100.0 if mp == 0 else 100.0 - 100.0 / (1.0 + mg / mp)

    out[periodo] = valor(media_g, media_p)
    for i in range(periodo + 1, len(c)):
        media_g = (media_g * (periodo - 1) + g[i - 1]) / periodo
        media_p = (media_p * (periodo - 1) + p[i - 1]) / periodo
        out[i] = valor(media_g, media_p)
    return pd.Series(out, index=cierre.index)


@dataclass
class Resultado:
    ticker: str
    fecha: date
    precio: float
    variacion: float | None
    rsi: float | None
    rsi_prev: float | None
    ma50: float | None
    ma200: float | None
    senales: list[str] = field(default_factory=list)   # señales según MODO
    debajo: dict = field(default_factory=dict)         # situación actual (para la prueba)


def _ok(*xs) -> bool:
    return all(x is not None and not np.isnan(x) for x in xs)


def analizar(serie: pd.Series, ticker: str, modo: str = MODO) -> Resultado | None:
    s = serie.dropna()
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < RSI_PERIODO + 2:
        return None
    rsi = calcular_rsi(s)
    ma50 = s.rolling(MA_RAPIDA).mean()
    ma200 = s.rolling(MA_LENTA).mean()
    c, c_prev = float(s.iloc[-1]), float(s.iloc[-2])

    def f(x):
        return None if pd.isna(x) else float(x)

    r = Resultado(
        ticker=ticker, fecha=s.index[-1].date(), precio=c,
        variacion=(c / c_prev - 1) * 100 if c_prev else None,
        rsi=f(rsi.iloc[-1]), rsi_prev=f(rsi.iloc[-2]),
        ma50=f(ma50.iloc[-1]), ma200=f(ma200.iloc[-1]),
    )
    niveles = {
        "rsi": (r.rsi, r.rsi_prev, RSI_UMBRAL, RSI_UMBRAL),
        "ma50": (c, c_prev, r.ma50, f(ma50.iloc[-2])),
        "ma200": (c, c_prev, r.ma200, f(ma200.iloc[-2])),
    }
    for nombre, (actual, previo, nivel, nivel_prev) in niveles.items():
        if not _ok(actual, nivel):
            continue                       # p. ej. acción reciente sin 200 días de historial
        r.debajo[nombre] = actual < nivel
        if modo == "estado":
            disparo = actual < nivel
        else:
            disparo = _ok(previo, nivel_prev) and previo >= nivel_prev and actual < nivel
        if disparo:
            r.senales.append(nombre)
    return r


# --------------------------------------------------------------- mensajes
def _detalle(r: Resultado, senal: str) -> str:
    verbo = "cruzó bajo" if MODO != "estado" else "bajo"
    if senal == "rsi":
        previo = f" (ayer {num(r.rsi_prev, 1)})" if r.rsi_prev is not None else ""
        return f"RSI {num(r.rsi, 1)}{previo} → {verbo} {RSI_UMBRAL:g}"
    nivel = r.ma50 if senal == "ma50" else r.ma200
    nombre = "MA50" if senal == "ma50" else "MA200"
    return f"{verbo} {nombre} ({num(nivel)}; {pct((r.precio / nivel - 1) * 100)})"


def componer_alerta(alertas: list[tuple[Resultado, list[str]]], ventana: str, ahora_ny: datetime,
                    vigiladas: int) -> tuple[list[str], str, str]:
    """Devuelve (bloques HTML para Telegram, texto plano para email, asunto)."""
    local = ahora_ny.astimezone(TZ_LOCAL)
    fecha = f"{DIAS_SEMANA[local.weekday()]} {local:%d/%m/%Y %H:%M} ({NOMBRE_TZ_LOCAL})"
    vela = max(r.fecha for r, _ in alertas).strftime("%d/%m")
    if ventana == "apertura":
        titulo, nota = "Apertura", f"⏳ Vela del {vela} en curso: señal provisional hasta el cierre."
    else:
        titulo, nota = "Cierre", f"✅ Vela del {vela} cerrada: señal confirmada."
    cabecera_txt = f"🔔 Nasdaq-100 · {titulo} · {fecha}\n{nota}"
    cabecera_html = f"🔔 <b>Nasdaq-100 · {titulo}</b> · {html.escape(fecha)}\n{nota}"

    orden = sorted(alertas, key=lambda a: (-len(a[1]), a[0].variacion or 0))
    bloques_html, bloques_txt = [cabecera_html], [cabecera_txt]
    for r, senales in orden:
        var = f" ({pct(r.variacion, 2)} en el día)" if r.variacion is not None else ""
        lineas = [f"  ↘ {_detalle(r, s)}" for s in senales]
        url = f"https://www.tradingview.com/chart/?symbol=NASDAQ%3A{urllib.parse.quote(r.ticker)}"
        bloques_html.append(f'📉 <a href="{url}"><b>{html.escape(r.ticker)}</b></a> · {num(r.precio)} USD{var}\n'
                            + html.escape("\n".join(lineas)))
        bloques_txt.append(f"📉 {r.ticker} · {num(r.precio)} USD{var}\n" + "\n".join(lineas))

    conteo = {s: sum(s in a[1] for a in alertas) for s in ETIQUETAS}
    resumen = " · ".join(f"{ETIQUETAS[s]}: {n}" for s, n in conteo.items() if n)
    pie = f"{len(alertas)} con señal de {vigiladas} vigiladas — {resumen}"
    bloques_html.append(html.escape(pie))
    bloques_txt.append(pie)
    asunto = f"🔔 Nasdaq-100 {titulo}: {len(alertas)} acciones con señal ({resumen})"
    return bloques_html, "\n\n".join(bloques_txt), asunto


def componer_prueba(resultados: list[Resultado], vigiladas: int, sin_datos: list[str],
                    ahora_ny: datetime) -> tuple[list[str], str, str]:
    ultima = max(r.fecha for r in resultados)
    hoy = ahora_ny.date()
    abajo = {s: [r.ticker for r in resultados if r.debajo.get(s)] for s in ETIQUETAS}
    cruces = [(r.ticker, r.senales) for r in resultados if r.senales]
    sin_ma200 = [r.ticker for r in resultados if r.ma200 is None]
    lineas = [
        "✅ Prueba del bot Nasdaq-100: funciona",
        f"Acciones con datos: {len(resultados)} de {vigiladas} · última vela {ultima:%d/%m/%Y}",
        f"Modo: {MODO} · RSI(14) < {RSI_UMBRAL:g} · MA50 · MA200 (velas de 1 día)",
        "",
        "Situación actual:",
        f"  • RSI < {RSI_UMBRAL:g}: {len(abajo['rsi'])}" + (f" ({', '.join(abajo['rsi'])})" if abajo["rsi"] else ""),
        f"  • Bajo MA50: {len(abajo['ma50'])}",
        f"  • Bajo MA200: {len(abajo['ma200'])}",
        "",
        "Señales en la última vela (lo que se habría avisado):",
    ]
    lineas += ([f"  • {t}: {', '.join(ETIQUETAS[s] for s in ss)}" for t, ss in cruces] or ["  • ninguna"])
    if sin_ma200:
        lineas += ["", f"Sin 200 días de historial (sin señal MA200): {', '.join(sin_ma200)}"]
    if sin_datos:
        lineas += ["", f"⚠️ Sin datos en Yahoo: {', '.join(sin_datos)}"]
    lineas += ["", f"Revisiones automáticas: apertura ~{hora_local(hoy, dtime(9, 45))} y cierre "
                   f"~{hora_local(hoy, dtime(16, 15))} hora de {NOMBRE_TZ_LOCAL}, días hábiles de NY."]
    texto = "\n".join(lineas)
    return [html.escape(b) for b in texto.split("\n\n")], texto, "✅ Prueba del bot Nasdaq-100"


def trocear(bloques: list[str], limite: int = 3800) -> list[str]:
    """Agrupa bloques en mensajes de menos de `limite` caracteres (Telegram admite 4096)."""
    mensajes, actual = [], ""
    for b in bloques:
        while len(b) > limite:                       # bloque gigante: se corta a la fuerza
            if actual:
                mensajes.append(actual)
                actual = ""
            mensajes.append(b[:limite])
            b = b[limite:]
        candidato = f"{actual}\n\n{b}" if actual else b
        if len(candidato) > limite:
            mensajes.append(actual)
            actual = b
        else:
            actual = candidato
    if actual:
        mensajes.append(actual)
    return mensajes


# ----------------------------------------------------------------- envío
def _telegram(token: str, chat_id: str, texto: str) -> None:
    datos = urllib.parse.urlencode({"chat_id": chat_id, "text": texto, "parse_mode": "HTML",
                                    "disable_web_page_preview": "true"}).encode()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for intento in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=datos), timeout=20) as resp:
                if json.load(resp).get("ok"):
                    return
        except urllib.error.HTTPError as e:
            cuerpo = e.read().decode(errors="replace")[:300]
            if e.code == 429:
                time.sleep(5 * (intento + 1))
                continue
            raise RuntimeError(f"Telegram rechazó el mensaje ({e.code}): {cuerpo}") from None
        except urllib.error.URLError as e:
            log.warning("Telegram no responde (%s), reintento", e.reason)
        time.sleep(3 * (intento + 1))
    raise RuntimeError("No pude enviar el mensaje a Telegram")


def _email_configurado() -> bool:
    return bool(_env("SMTP_USER") and _env("SMTP_PASS"))


def _email(asunto: str, texto: str) -> None:
    """Envía por SMTP. Por defecto usa Gmail y se envía el correo a sí mismo."""
    usuario = _env("CAAGUAZU001@GMAIL.COM")
    destinos = [d.strip() for d in _env("EMAIL_TO", usuario).split(",") if d.strip()]
    msg = EmailMessage()
    msg["Subject"] = asunto
    msg["From"] = _env("EMAIL_FROM", usuario)
    msg["To"] = ", ".join(destinos)
    msg.set_content(texto)
    host, puerto = _env("SMTP_HOST", "smtp.gmail.com"), int(_env("SMTP_PORT", "587"))
    if puerto == 465:
        servidor = smtplib.SMTP_SSL(host, puerto, timeout=30)
    else:
        servidor = smtplib.SMTP(host, puerto, timeout=30)
        servidor.starttls()
    with servidor:
        servidor.login(usuario, _env("dkqx xuwo hsrt uruv").replace(" ", ""))
        servidor.send_message(msg)


def enviar(bloques_html: list[str], texto: str, asunto: str) -> list[str]:
    """Envía por email (principal) y por Telegram si también está configurado."""
    canales, errores = [], []
    if _email_configurado():
        try:
            _email(asunto, texto)
            canales.append("email")
        except Exception as e:
            errores.append(f"Email: {e}")
    token, chats = _env("TELEGRAM_TOKEN"), _env("TELEGRAM_CHAT_ID")
    if token and chats:
        try:
            for chat in [c.strip() for c in chats.split(",") if c.strip()]:
                for trozo in trocear(bloques_html):
                    _telegram(token, chat, trozo)
            canales.append("telegram")
        except Exception as e:
            errores.append(f"Telegram: {e}")
    for err in errores:
        log.error(err)
    if not canales:
        print(texto)
        raise RuntimeError("Ningún canal de aviso funcionó. " + " | ".join(errores)
                           if errores else "No hay canal de aviso configurado (SMTP_USER y SMTP_PASS)")
    return canales


def _resumen_github(texto: str) -> None:
    ruta = os.getenv("GITHUB_STEP_SUMMARY")
    if ruta:
        with open(ruta, "a", encoding="utf-8") as fh:
            fh.write(texto + "\n")


# ------------------------------------------------------------- ejecución
def ejecutar(ventana: str = "auto", ahora: datetime | None = None, seco: bool = False,
             descargar: Callable[[list[str]], pd.DataFrame] = descargar_cierres,
             notificar: Callable[..., list[str]] = enviar, espera_reintento: int = 90) -> int:
    ahora_ny = (ahora or datetime.now(NY)).astimezone(NY)
    hoy = ahora_ny.date()
    estado = cargar_estado()
    automatico = ventana == "auto"
    if seco:
        notificar = lambda b, t, a: (print(t), ["consola"])[1]  # noqa: E731

    if automatico:
        ventana = decidir_ventana(ahora_ny, estado)
        if ventana is None:
            log.info("%s NY: no toca revisar (fuera de ventana, feriado o ya hecho).",
                     ahora_ny.strftime("%a %d/%m %H:%M"))
            return 0
    prueba = ventana == "prueba"
    # Solo las revisiones automáticas marcan la ventana como hecha: una ejecución manual
    # no debe impedir la revisión programada de ese día.
    registro = (estado.setdefault("ventanas", {}).setdefault(hoy.isoformat(), {})
                .setdefault(ventana, {"intentos": 0})) if automatico else {}
    log.info("Revisión: %s (%s NY)", ventana, ahora_ny.strftime("%d/%m/%Y %H:%M"))

    try:
        tickers = leer_tickers()
        for vuelta in range(2):
            cierres = descargar(tickers)
            resultados, sin_datos, viejos = [], [], []
            for t in tickers:
                r = analizar(cierres[t], t) if t in cierres.columns else None
                if r is None:
                    sin_datos.append(t)
                elif automatico and r.fecha != hoy:
                    viejos.append(t)          # Yahoo aún no tiene la vela de hoy
                else:
                    resultados.append(r)
            if prueba or len(resultados) >= 0.5 * len(tickers) or vuelta == 1:
                break
            log.warning("Solo %d/%d acciones tienen la vela de hoy; espero %ds y reintento",
                        len(resultados), len(tickers), espera_reintento)
            time.sleep(espera_reintento)

        if not resultados:
            raise DatosIncompletos("Yahoo Finance no devolvió datos utilizables")
        if automatico and len(resultados) < 0.5 * len(tickers):
            raise DatosIncompletos(f"Solo {len(resultados)} de {len(tickers)} acciones tienen la vela "
                                   f"de hoy ({hoy:%d/%m/%Y}); ¿mercado cerrado o datos atrasados?")
        if sin_datos:
            log.warning("Sin datos: %s", ", ".join(sin_datos))

        if prueba:
            notificar(*componer_prueba(resultados, len(tickers), sin_datos, ahora_ny))
            log.info("Mensaje de prueba enviado")
            return 0

        enviadas = estado.setdefault("enviadas", {})
        alertas = []
        for r in resultados:
            nuevas = [s for s in r.senales if f"{r.ticker}|{s}|{r.fecha.isoformat()}" not in enviadas]
            if nuevas:
                alertas.append((r, nuevas))

        canales = []
        if alertas:
            canales = notificar(*componer_alerta(alertas, ventana, ahora_ny, len(tickers)))
            for r, nuevas in alertas:
                for s in nuevas:
                    enviadas[f"{r.ticker}|{s}|{r.fecha.isoformat()}"] = ahora_ny.isoformat(timespec="minutes")
        registro.update(estado="ok", hora=ahora_ny.strftime("%H:%M"), alertas=len(alertas))
        registro.pop("error", None)
        if not seco:
            guardar_estado(estado, hoy)
        msg = (f"{ventana}: {len(resultados)} acciones analizadas, {len(alertas)} con señal nueva"
               + (f" → enviado por {', '.join(canales)}" if canales else " → sin avisos"))
        log.info(msg)
        _resumen_github(f"**{msg}**" + ("".join(f"\n- {r.ticker}: {', '.join(ETIQUETAS[s] for s in ss)}"
                                                for r, ss in alertas)))
        return 0

    except Exception as e:
        log.exception("La revisión falló")
        if not automatico or seco:
            return 1
        registro["intentos"] = registro.get("intentos", 0) + 1
        registro["error"] = str(e)[:300]
        if registro["intentos"] >= MAX_INTENTOS:
            registro["estado"] = "agotada"
            aviso = (f"⚠️ Bot Nasdaq-100: la revisión de {ventana} del {hoy:%d/%m/%Y} falló "
                     f"{registro['intentos']} veces.\nÚltimo error: {e}")
            try:
                notificar([html.escape(aviso)], aviso, "⚠️ Bot Nasdaq-100: revisión fallida")
            except Exception:
                log.error("Tampoco pude avisar del error")
        guardar_estado(estado, hoy)
        # El primer fallo no marca la ejecución como fallida en GitHub (evita emails por un
        # retraso puntual de Yahoo): la ejecución de respaldo lo vuelve a intentar.
        return 1 if registro.get("estado") == "agotada" else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    p = argparse.ArgumentParser(description="Bot de vigilancia Nasdaq-100 (RSI 35, MA50, MA200)")
    p.add_argument("--ventana", default="auto", choices=["auto", "apertura", "cierre", "prueba"])
    p.add_argument("--seco", action="store_true", help="no envía ni guarda; imprime el mensaje")
    a = p.parse_args()
    return ejecutar(a.ventana, seco=a.seco)


if __name__ == "__main__":
    sys.exit(main())
