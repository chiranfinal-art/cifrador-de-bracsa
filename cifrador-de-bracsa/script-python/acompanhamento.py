#!/usr/bin/env python3
"""
acompanhamento.py - ouve uma gravação de violino e sugere acordes para brácsa e bőgő.

Uso básico:
    python acompanhamento.py minha_musica.wav

Gera, ao lado do arquivo de entrada:
    minha_musica_acordes.txt     cifra compasso a compasso (acorde, notas do brácsa, nota do bőgő)
    minha_musica_acomp.mid       MIDI do acompanhamento, alinhado com a gravação
    minha_musica_previa.wav      sua gravação + acompanhamento sintetizado, para conferir de ouvido

Duas formas de análise (escolhidas automaticamente, ou com --modo):
    compasso  1 acorde a cada N tempos; para música com pulso regular
    livre     os acordes seguem as mudanças da melodia, sem depender do pulso;
              para rubato, arco muito ligado ou andamento que varia

Opções úteis (veja todas com --help):
    --de 1:10 --ate 2:05    analisa só um trecho (ex.: uma das músicas da gravação)
    --modo livre            força o modo livre
    --dur-min 1.5           modo livre: duração mínima de cada acorde, em segundos
    --bpm 120               informa o andamento (se a detecção automática errar)
    --tempos-por-acorde 2   quantos tempos cada acorde dura no mínimo (padrão 2)
    --tempos-por-compasso 2 só para numerar os compassos na cifra (padrão 2)
    --inicio 0.8            segundo em que cai o primeiro tempo forte
    --estabilidade 0.8      maior = menos trocas de acorde
    --sem-setima            escreve só tríades (A em vez de A7)
    --tonalidade Am         força a tonalidade, se a detectada estiver errada

Requisitos: Python 3.9+, numpy, scipy.
Para mp3/m4a/ogg é preciso ter o ffmpeg instalado; WAV funciona sem ele.
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

SR = 22050
N_FFT = 4096
HOP = 512

NOMES_SUST = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOMES_BEMOL = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NOMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NOMES_PT = ["Dó", "Dó#", "Ré", "Mib", "Mi", "Fá", "Fá#", "Sol", "Láb", "Lá", "Sib", "Si"]


def nome_para_classe(nome):
    nome = nome.strip()
    for lista in (NOMES_SUST, NOMES_BEMOL):
        if nome in lista:
            return lista.index(nome)
    raise ValueError(f"nota desconhecida: {nome}")


def definir_grafia(tonica, modo):
    """Usa sustenidos ou bemóis conforme a armadura da tonalidade
    (em Lá menor a sensível é G#, não Ab)."""
    rel_maior = tonica if modo == "maior" else (tonica + 3) % 12
    bemois = rel_maior in (5, 10, 3, 8, 1)  # F, Bb, Eb, Ab, Db
    NOMES[:] = NOMES_BEMOL if bemois else NOMES_SUST


# ----------------------------------------------------------------------------
# 1. Leitura do áudio
# ----------------------------------------------------------------------------
def carregar_audio(caminho):
    """Devolve sinal mono float32 em SR Hz."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-v", "error", "-i", caminho, "-f", "f32le",
               "-ac", "1", "-ar", str(SR), "-"]
        bruto = subprocess.run(cmd, capture_output=True, check=True).stdout
        y = np.frombuffer(bruto, dtype=np.float32).copy()
    else:
        if not caminho.lower().endswith(".wav"):
            sys.exit("Sem ffmpeg instalado só consigo abrir .wav. "
                     "Exporte a gravação como WAV ou instale o ffmpeg.")
        sr, y = wavfile.read(caminho)
        if y.dtype.kind == "i":
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        elif y.dtype.kind == "u":
            y = (y.astype(np.float32) - 128) / 128
        y = y.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, SR)
            y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    pico = np.max(np.abs(y)) if len(y) else 0
    if pico > 0:
        y = y / pico
    return y


def stft_mag(y, n_fft=N_FFT):
    janela = np.hanning(n_fft).astype(np.float32)
    y = np.pad(y, n_fft // 2)
    n = 1 + (len(y) - n_fft) // HOP
    quadros = np.lib.stride_tricks.as_strided(
        y, shape=(n, n_fft), strides=(y.strides[0] * HOP, y.strides[0]))
    return np.abs(np.fft.rfft(quadros * janela, axis=1)).astype(np.float32)  # (quadros, bins)


# ----------------------------------------------------------------------------
# 2. Quais notas estão soando (saliência harmônica)
# ----------------------------------------------------------------------------
def estimar_afinacao(y):
    """Lá de referência do violino (Hz), pelo desvio médio das notas em relação a 440."""
    n = 8192
    janela = np.hanning(n)
    desvios, pesos = [], []
    lo, hi = int(180 * n / SR), int(2000 * n / SR)
    for i in range(0, len(y) - n, 2048):
        X = np.abs(np.fft.rfft(y[i:i + n] * janela))
        k = lo + int(np.argmax(X[lo:hi]))
        a, b, c = np.log(X[k - 1:k + 2] + 1e-9)
        den = a - 2 * b + c
        if den >= 0:
            continue
        f = (k + 0.5 * (a - c) / den) * SR / n
        m = 69 + 12 * np.log2(f / 440)
        desvios.append(m - round(m))
        pesos.append(X[k])
    if len(desvios) < 20:
        return 440.0
    d, w = np.array(desvios), np.array(pesos)
    d = d[w > np.percentile(w, 40)]
    # média circular (o desvio 'dá a volta' em ±50 cents)
    ang = np.angle(np.mean(np.exp(2j * np.pi * d)))
    return 440.0 * 2 ** (ang / (2 * np.pi) / 12)


def saliencia_notas(S, nota_min=52, nota_max=100, n_harm=6, ref=440.0, n_fft=None):
    """Para cada quadro, uma 'força' para cada nota MIDI, somando seus harmônicos.
    Exige que a fundamental esteja presente, para não confundir uma nota com a
    oitava de baixo (erro clássico em violino)."""
    freq_bin = SR / (n_fft or N_FFT)
    logS = np.log1p(100 * S / (S.max() + 1e-9))
    notas = np.arange(nota_min, nota_max + 1)
    sal = np.zeros((S.shape[0], len(notas)), dtype=np.float32)
    nb = S.shape[1]
    for i, m in enumerate(notas):
        f0 = ref * 2 ** ((m - 69) / 12)
        total = np.zeros(S.shape[0], dtype=np.float32)
        fund = None
        for h in range(1, n_harm + 1):
            b = int(round(h * f0 / freq_bin))
            if b + 1 >= nb:
                break
            mag = logS[:, b - 1:b + 2].max(axis=1)
            if h == 1:
                fund = mag
            total += (0.8 ** (h - 1)) * mag
        sal[:, i] = total * np.clip(fund / (0.35 * total / n_harm + 1e-6), 0, 1)
    # mantém só as notas mais fortes de cada quadro
    limiar = 0.45 * sal.max(axis=1, keepdims=True)
    sal[sal < limiar] = 0
    # remove 'notas fantasmas' que são só harmônicos de uma nota mais grave
    # (oitava, oitava+quinta, 2 oitavas, 2 oitavas+terça, 2 oitavas+quinta)
    harm = (12, 19, 24, 28, 31)
    for q in np.nonzero(sal.max(axis=1) > 0)[0]:
        linha = sal[q]
        ativos = np.nonzero(linha)[0]
        for j in ativos:  # do grave para o agudo
            if linha[j] == 0:
                continue
            for d in harm:
                k = j + d
                if k < len(linha) and 0 < linha[k] < 1.15 * linha[j]:
                    linha[k] = 0
    # silêncio: zera quadros muito fracos
    energia = logS.sum(axis=1)
    sal[energia < 0.08 * np.percentile(energia, 95)] = 0
    return notas, sal


def croma_de_notas(notas, sal):
    croma = np.zeros((sal.shape[0], 12), dtype=np.float32)
    for i, m in enumerate(notas):
        croma[:, m % 12] += sal[:, i]
    return croma


# ----------------------------------------------------------------------------
# 3. Pulso (tempo e batidas)
# ----------------------------------------------------------------------------
def curva_de_ataques(S_curto, sal):
    logS = np.log1p(S_curto[:sal.shape[0]])
    fluxo = np.maximum(0, np.diff(logS, axis=0)).sum(axis=1)
    # mudança de nota também conta como ataque (arcos ligados quase não têm 'ataque')
    mud = np.abs(np.diff(sal, axis=0)).sum(axis=1)
    def norm(x):
        x = x - uniform_filter1d(x, 16)
        x = np.maximum(x, 0)
        return x / (x.std() + 1e-9)
    env = norm(fluxo) + norm(mud)
    return np.concatenate([[0], env])


def estimar_bpm(env, bpm_min=60, bpm_max=200):
    fps = SR / HOP
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    bpms = 60 * fps / np.maximum(lags, 1)
    ok = (bpms >= bpm_min) & (bpms <= bpm_max)
    # preferência suave por andamentos ao redor de 120
    peso = np.exp(-0.5 * (np.log2(bpms / 120) / 0.9) ** 2)
    # o tempo costuma se dividir em 2 (colcheias): premia pulsos cuja metade
    # também é regular, para não confundir o tempo com grupos de 3 colcheias
    metade = np.interp(lags / 2, lags, ac)
    score = np.where(ok, (ac + 0.5 * metade) * peso, -np.inf)
    lag = int(np.argmax(score))
    # interpolação parabólica: precisão abaixo de um quadro
    if 1 <= lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        den = a - 2 * b + c
        if den < 0:
            lag = lag + 0.5 * (a - c) / den
    return 60 * fps / lag


def rastrear_batidas(env, bpm, rigidez=100):
    """Programação dinâmica de Ellis (2007)."""
    fps = SR / HOP
    periodo = 60 * fps / bpm
    env = env / (env.std() + 1e-9)
    n = len(env)
    score = env.copy()
    anterior = -np.ones(n, dtype=int)
    janela = np.arange(int(-2 * periodo), -int(periodo / 2) + 1)
    custo = -rigidez * (np.log(-janela / periodo) ** 2)
    for t in range(n):
        idx = t + janela
        v = idx >= 0
        if not v.any():
            continue
        cand = score[idx[v]] + custo[v]
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[t] = env[t] + cand[k]
            anterior[t] = idx[v][k]
    # começa do melhor ponto perto do fim
    fim = len(score) - int(periodo)
    t = int(np.argmax(score[max(0, fim):])) + max(0, fim)
    batidas = []
    while t >= 0:
        batidas.append(t)
        t = anterior[t]
    batidas = np.array(batidas[::-1])
    return batidas / fps


def grade_fixa(bpm, inicio, duracao):
    passo = 60.0 / bpm
    return np.arange(inicio, duracao, passo)


# ----------------------------------------------------------------------------
# 4. Tonalidade
# ----------------------------------------------------------------------------
PERFIL_MAIOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def candidatos_tonalidade(croma_total):
    """Tonalidades ordenadas pela semelhança com o perfil de notas (Krumhansl)."""
    res = []
    for tonica in range(12):
        for modo, perfil in (("maior", PERFIL_MAIOR), ("menor", PERFIL_MENOR)):
            res.append((np.corrcoef(croma_total, np.roll(perfil, tonica))[0, 1], tonica, modo))
    return sorted(res, reverse=True)


def estimar_tonalidade(croma_total, croma=None, n_cand=4):
    """Pega as tonalidades mais prováveis pelo perfil de notas e desempata
    harmonizando a música em cada uma: fica a que explica melhor as notas
    com acordes 'da casa'."""
    cands = candidatos_tonalidade(croma_total)
    if croma is None:
        return cands[0][1], cands[0][2]
    vocab = vocabulario(True)
    passo = int(SR / HOP)  # fatias de ~1 s
    perfis = [croma[i:i + passo].sum(axis=0) for i in range(0, croma.shape[0], passo)]
    perfis = [p / p.sum() for p in perfis if p.sum() > 0]
    melhor = None
    for r, tonica, modo in cands[:n_cand]:
        fit = np.mean([max(pontuar(a, p) + bonus_tonal(a, tonica, modo) for a in vocab)
                       for p in perfis])
        nota = fit + 0.3 * r
        if melhor is None or nota > melhor[0]:
            melhor = (nota, tonica, modo)
    return melhor[1], melhor[2]


# ----------------------------------------------------------------------------
# 5. Acordes
# ----------------------------------------------------------------------------
class Acorde:
    def __init__(self, raiz, tipo):
        self.raiz, self.tipo = raiz, tipo
        intervalos = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10)}[tipo]
        self.notas = [(raiz + i) % 12 for i in intervalos]
        # pesos: fundamental e terça definem o acorde; quinta e sétima pesam menos
        self.pesos = {"": (1.1, 1.0, 0.8), "m": (1.1, 1.0, 0.8),
                      "7": (1.1, 1.0, 0.7, 0.6)}[tipo]

    @property
    def nome(self):
        return NOMES[self.raiz] + self.tipo

    def grafia(self):
        """Nomes das notas com a grafia correta (terça de E é G#, não Ab)."""
        letras = "CDEFGAB"
        naturais = [0, 2, 4, 5, 7, 9, 11]
        nome_raiz = NOMES[self.raiz]
        li = letras.index(nome_raiz[0])
        saida = [nome_raiz]
        for n, passo in zip(self.notas[1:], (2, 4, 6)):
            letra = letras[(li + passo) % 7]
            dif = (n - naturais[letras.index(letra)] + 6) % 12 - 6
            saida.append(letra + {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[dif])
        return saida

    def __repr__(self):
        return self.nome


def vocabulario(com_setima=True):
    tipos = ["", "m"] + (["7"] if com_setima else [])
    return [Acorde(r, t) for t in tipos for r in range(12)]


def pontuar(acorde, perfil):
    """perfil: vetor de 12 classes de altura, soma 1."""
    dentro = sum(w * perfil[n] for n, w in zip(acorde.notas, acorde.pesos))
    fora = 1.0 - sum(perfil[n] for n in acorde.notas)
    s = dentro - 0.6 * fora
    if acorde.tipo == "7":
        s -= 0.08  # só compensa a tétrade quando a sétima soa de verdade
    return s


def bonus_tonal(acorde, tonica, modo):
    grau = (acorde.raiz - tonica) % 12
    if modo == "maior":
        diat = {0: {"": .10}, 2: {"m": .10}, 4: {"m": .10}, 5: {"": .10},
                7: {"": .10, "7": .10}, 9: {"m": .10}}
    else:
        # menor natural + harmônica (V, V7); IV maior (dórico) e v menor valem um
        # pouco menos, para só aparecerem quando a melodia pedir (F# ou G natural)
        diat = {0: {"m": .10}, 3: {"": .10}, 5: {"m": .10, "": .06},
                7: {"": .10, "7": .10, "m": .05}, 8: {"": .10}, 10: {"": .10}}
    b = diat.get(grau, {}).get(acorde.tipo)
    if b is not None:
        return b
    if acorde.tipo in ("7", ""):  # dominantes secundárias (V de algum grau da tonalidade)
        alvo = (acorde.raiz + 5) % 12
        if (alvo - tonica) % 12 in diat:
            return 0.05 if acorde.tipo == "7" else 0.02
    # acorde estranho à tonalidade: só aparece se a melodia insistir muito nele
    return -0.06


def escolher_acordes(perfis, vocab, tonica, modo, estabilidade):
    """Viterbi: melhor sequência de acordes, penalizando trocas desnecessárias."""
    n, k = len(perfis), len(vocab)
    ganho = np.array([[pontuar(a, p) + bonus_tonal(a, tonica, modo) for a in vocab]
                      for p in perfis])
    vazio = np.array([p.sum() < 1e-6 for p in perfis])
    ganho[vazio] = 0  # trechos de silêncio: qualquer acorde, mantém o anterior
    melhor = ganho[0].copy()
    volta = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        trocar = melhor.max() - estabilidade
        idx_max = int(np.argmax(melhor))
        ficar = melhor
        escolha_ficar = ficar >= trocar
        volta[t] = np.where(escolha_ficar, np.arange(k), idx_max)
        melhor = np.maximum(ficar, trocar) + ganho[t]
    seq = [int(np.argmax(melhor))]
    for t in range(n - 1, 0, -1):
        seq.append(volta[t, seq[-1]])
    seq = seq[::-1]
    confianca = []
    for t, j in enumerate(seq):
        g = np.sort(ganho[t])[::-1]
        confianca.append(0.0 if vazio[t] else float(ganho[t, j] - g[1] if ganho[t, j] == g[0] else ganho[t, j] - g[0]))
    return [vocab[j] for j in seq], confianca, vazio, float(melhor.max())


def segmentar_livre(croma, tonica, modo, vocab, troca=0.6, dur_min=1.0, fatia=0.25):
    """Acordes SEM depender do pulso: divide em fatias curtas e escolhe onde a
    harmonia muda (Viterbi), exigindo que cada acorde dure pelo menos dur_min s.
    Bom para música com rubato ou arco muito ligado.
    Devolve lista de (inicio_s, fim_s, acorde, certeza)."""
    fps = SR / HOP
    passo = max(1, int(round(fatia * fps)))
    marcos = np.arange(0, croma.shape[0], passo)
    perf = np.array([croma[i:i + passo].sum(axis=0) for i in marcos])
    energia = perf.sum(axis=1)
    if not (energia > 0).any():
        return []
    # cada fatia pesa conforme o quanto soa: pausas não decidem nada
    peso = np.clip(energia / (np.percentile(energia[energia > 0], 75) + 1e-9), 0, 1)
    pn = perf / np.maximum(energia[:, None], 1e-9)
    G = np.array([[pontuar(a, p) + bonus_tonal(a, tonica, modo) for a in vocab]
                  for p in pn]) * peso[:, None]
    n, k = G.shape
    D = max(1, int(round(dur_min / fatia)))
    NEG = -1e18
    # estado (acorde j, há d fatias); d == D significa 'já cumpriu a duração mínima'
    sc = np.full((k, D + 1), NEG)
    sc[:, 1 if D > 1 else D] = G[0]
    volta = np.zeros((n, k, D + 1, 2), dtype=np.int32)
    idx = np.arange(k)
    for t in range(1, n):
        novo = np.full((k, D + 1), NEG)
        for d in range(1, D + 1):
            dn = min(d + 1, D)
            cand = sc[:, d] + G[t]
            m = cand > novo[:, dn]
            novo[m, dn] = cand[m]
            volta[t, m, dn, 0] = idx[m]
            volta[t, m, dn, 1] = d
        jb = int(np.argmax(sc[:, D]))
        cand = sc[jb, D] - troca + G[t]
        d1 = 1 if D > 1 else D
        m = (cand > novo[:, d1]) & (idx != jb)
        novo[m, d1] = cand[m]
        volta[t, m, d1, 0] = jb
        volta[t, m, d1, 1] = D
        sc = novo
    j, d = np.unravel_index(int(np.argmax(sc)), sc.shape)
    seq = [int(j)]
    for t in range(n - 1, 0, -1):
        j, d = volta[t, j, d]
        seq.append(int(j))
    seq = seq[::-1]
    segs = []
    for i, j in enumerate(seq):
        a = marcos[i] / fps
        b = min(croma.shape[0], marcos[i] + passo) / fps
        if segs and segs[-1][2] is vocab[j]:
            segs[-1][1] = b
        else:
            segs.append([a, b, vocab[j]])
    # certeza: quanto o acorde escolhido ganha do 2º melhor no trecho inteiro
    tq = np.arange(croma.shape[0]) / fps
    saida = []
    for a, b, ac in segs:
        p = croma[(tq >= a) & (tq < b)].sum(axis=0)
        if p.sum() == 0:
            saida.append((a, b, ac, None))
            continue
        p = p / p.sum()
        g = sorted((pontuar(v, p) + bonus_tonal(v, tonica, modo) for v in vocab), reverse=True)
        meu = pontuar(ac, p) + bonus_tonal(ac, tonica, modo)
        saida.append((a, b, ac, float(meu - (g[1] if meu >= g[0] else g[0]))))
    return saida


def pulso_estavel(env, bpm):
    """O andamento se mantém ao longo da gravação? (janelas de 10 s)"""
    fps = SR / HOP
    W = int(10 * fps)
    if len(env) < 2 * W:
        return True
    ok = tot = 0
    for t in range(0, len(env) - W, int(5 * fps)):
        b = estimar_bpm(env[t:t + W])
        desvio = abs(np.log2(b / bpm) - round(np.log2(b / bpm)))  # aceita dobro/metade
        ok += desvio < 0.07
        tot += 1
    return ok / tot >= 0.6


def melodia_trecho(notas, sal, a, b, max_notas=12):
    """Notas principais do violino entre a e b (segundos), sem repetições seguidas."""
    fps = SR / HOP
    i0, i1 = int(a * fps), int(b * fps)
    passo = max(1, int(0.12 * fps))
    saida = []
    for i in range(i0, i1, passo):
        q = sal[i:min(i + passo, i1)].sum(axis=0)
        if q.max() <= 0:
            continue
        nm = NOMES[notas[int(np.argmax(q))] % 12]
        if not saida or saida[-1] != nm:
            saida.append(nm)
    if len(saida) > max_notas:
        saida = saida[:max_notas] + ["…"]
    return " ".join(saida)


# ----------------------------------------------------------------------------
# 6. Versos: transcrição em notas, arranques (felfutás/lefutás) e divisão em versos
# ----------------------------------------------------------------------------
def transcrever(y, ref, n_fft=2048, dur_min=0.06):
    """Linha melódica do violino em notas [inicio_s, fim_s, nota_midi]."""
    S = stft_mag(y, n_fft)
    notas, sal = saliencia_notas(S, nota_min=55, ref=ref, n_fft=n_fft)
    fps = SR / HOP
    top = np.where(sal.max(1) > 0, notas[np.argmax(sal, 1)], -1)
    # filtro de moda em 5 quadros (~0,12 s): tira tremidos de 1-2 quadros
    sm = top.copy()
    for i in range(2, len(top) - 2):
        w = top[i - 2:i + 3]
        w = w[w >= 0]
        if len(w):
            vals, cnt = np.unique(w, return_counts=True)
            sm[i] = vals[np.argmax(cnt)] if cnt.max() > 2 else top[i]
    # ataques: nova arcada na mesma nota também começa nota nova
    env = curva_de_ataques(stft_mag(y, 1024), sal)
    lim = np.percentile(env, 92)
    picos = set(i for i in range(1, len(env) - 1)
                if env[i] > lim and env[i] >= env[i - 1] and env[i] >= env[i + 1])
    res = []
    ini = 0
    for i in range(1, len(sm) + 1):
        if i == len(sm) or sm[i] != sm[ini] or (i in picos and i - ini > 3):
            if sm[ini] >= 0 and (i - ini) / fps >= dur_min:
                res.append([ini / fps, i / fps, int(sm[ini])])
            elif sm[ini] >= 0 and res and res[-1][2] == sm[ini]:
                res[-1][1] = i / fps
            ini = i
    out = []
    for n in res:
        if out and out[-1][2] == n[2] and n[0] - out[-1][1] < 0.05 and int(n[0] * fps) not in picos:
            out[-1][1] = n[1]
        else:
            out.append(n)
    return out


def forca_fim_de_frase(nts, k, mediana):
    """0..1: quanto a nota k-1 (mais o respiro depois dela) soa como fim de frase.
    Nota típica = 1x a mediana; fim de frase = 3x ou mais."""
    if k == 0:
        return 1.0
    ant = nts[k - 1]
    efetiva = (ant[1] - ant[0]) + 1.5 * max(0.0, nts[k][0] - ant[1])
    return float(np.clip((efetiva / mediana - 2.0) / 1.5, 0, 1))


def arranques(nts):
    """Arranque (felfutás/lefutás/volteio): 2-4 notas curtas logo depois de um fim de
    frase, levando a uma nota mais longa. Devolve dict: índice da nota-alvo ->
    (tipo, força, índice da 1ª nota do arranque)."""
    durs = np.array([n[1] - n[0] for n in nts])
    mediana = float(np.median(durs))
    res = {}
    for tam in (3, 2, 4):
        for i in range(len(nts) - tam):
            r = nts[i:i + tam]
            alvo = nts[i + tam]
            d = durs[i:i + tam]
            dalvo = durs[i + tam]
            if d.max() > 0.4 or dalvo < 1.2 * d.mean() or dalvo < 0.25:
                continue
            gaps = [r[q + 1][0] - r[q][1] for q in range(tam - 1)] + [alvo[0] - r[-1][1]]
            if max(gaps) > 0.12:
                continue
            ff = forca_fim_de_frase(nts, i, mediana)
            if ff < 0.3:
                continue  # arranque de verdade vem depois de um fim de frase
            alturas = [n[2] for n in r] + [alvo[2]]
            if any(alturas[q] == alturas[q + 2] for q in range(len(alturas) - 2)):
                continue  # trinado/bordadura (a-b-a): ornamento, não arranque
            passos = np.diff(alturas)
            if np.abs(passos).max() > 9:
                continue
            if np.all(passos > 0):
                tipo, f = "felfutás", 1.0
            elif np.all(passos < 0):
                tipo, f = "lefutás", 1.0
            else:
                tipo, f = "volteio", 0.7
            f *= {3: 1.0, 2: 0.6, 4: 0.8}[tam]
            f += 0.3 * min(1.0, dalvo / (2 * d.mean()))
            f *= 0.5 + 0.5 * ff
            k = i + tam
            if k not in res or res[k][1] < f:
                res[k] = (tipo, f, i)
    return res


def grade_alturas(nts, dur, dt=0.05):
    g = np.full(int(dur / dt) + 1, -1)
    for s, e, m in nts:
        g[int(s / dt):int(e / dt)] = m
    return g


def periodo_repeticao(nts, dur, lo=4.0, hi=20.0, dt=0.05):
    """Duração típica do verso: atraso em que a melodia mais se repete."""
    g = grade_alturas(nts, dur, dt)
    melhores = []
    for lag in range(int(lo / dt), min(int(hi / dt), len(g) // 2)):
        u, v = g[:-lag], g[lag:]
        ok = (u >= 0) & (v >= 0)
        melhores.append((((u == v) & ok).sum() / max(ok.sum(), 1), lag * dt))
    if not melhores:
        return None, 0
    melhores.sort(reverse=True)
    sc, L = melhores[0]
    # se a metade também repete quase tão bem, o verso é a metade
    for sc2, L2 in melhores:
        if abs(L2 - L / 2) < 0.6 and sc2 > 0.8 * sc:
            return L2, sc2
    return L, sc


def similaridade_local(g, t, L, dt=0.05, janela=3.0):
    """A melodia que começa em t reaparece ~L segundos antes ou depois?"""
    i, w = int(t / dt), int(janela / dt)
    melhor = 0
    for sinal in (-1, 1):
        for d in range(int(0.8 * L / dt), int(1.25 * L / dt)):
            j = i + sinal * d
            if j < 0 or j + w > len(g) or i + w > len(g):
                continue
            u, v = g[i:i + w], g[j:j + w]
            ok = (u >= 0) & (v >= 0)
            if ok.sum() < w / 3:
                continue
            melhor = max(melhor, ((u == v) & ok).sum() / ok.sum())
    return melhor


def achar_versos(nts, dur, L, beta=6.0):
    """Escolhe inícios de verso por programação dinâmica."""
    arr = arranques(nts)
    g = grade_alturas(nts, dur)
    durs = np.array([n[1] - n[0] for n in nts])
    mediana = float(np.median(durs))
    # início da música: pula notas soltas do começo (ruído, resto de outra música)
    p0 = 0
    while p0 < len(nts) - 1 and nts[p0 + 1][0] - nts[p0][1] > 0.3:
        p0 += 1
    # ... e, se a música começa com arranque, o verso 1 começa na nota de chegada
    k0 = p0
    for k, (tipo, f, i) in sorted(arr.items()):
        if i >= p0 and nts[i][0] - nts[p0][0] < 1.0:
            k0 = k
            break
    cand = []  # (tempo, pontuação, índice da nota, info do arranque)
    for k, n in enumerate(nts):
        if k <= k0:
            continue
        sc = 0.0
        info = arr.get(k)
        if info:
            sc += 1.0 + info[1]
        else:
            sc += 0.8 * forca_fim_de_frase(nts, k, mediana)
        if sc > 0.2:
            sc += 0.8 * similaridade_local(g, n[0], L)
            cand.append((n[0], sc, k, info))
    fim = nts[-1][1]
    # fim da música: a última nota longa (fim de frase) antes de restos/ruído do final
    for k in range(len(nts) - 1, max(k0, len(nts) - 25), -1):
        dk = durs[k]
        depois = nts[k + 1][0] - nts[k][1] if k + 1 < len(nts) else 9
        if dk >= 2.5 * mediana and (depois > 0.25 or k == len(nts) - 1):
            fim = nts[k][1]
            break
    cand = [c for c in cand if c[0] < fim - 0.5]
    pts = [(nts[k0][0], 0.0, k0, arr.get(k0))] + cand + [(fim, 0.0, len(nts), None)]
    n = len(pts)
    best = np.full(n, -1e18)
    prev = np.zeros(n, dtype=int)
    best[0] = 0
    for j in range(1, n):
        for i in range(j):
            d = pts[j][0] - pts[i][0]
            ultimo = j == n - 1
            if d > 1.6 * L or (d < 0.55 * L and not ultimo):
                continue
            custo = beta * np.log2(d / L) ** 2
            if ultimo and d < L:
                custo *= 0.3  # o final pode ser mais curto
            v = best[i] + pts[j][1] - custo
            if v > best[j]:
                best[j], prev[j] = v, i
    seq = [n - 1]
    while seq[-1] != 0:
        seq.append(prev[seq[-1]])
    seq = seq[::-1]
    res = [pts[i] for i in seq]
    # último verso bem mais longo que os outros = sobrou ruído/resto no final:
    # ele fica com a duração típica dos versos anteriores
    if len(res) >= 4:
        tipicos = np.median([res[i + 1][0] - res[i][0] for i in range(len(res) - 2)])
        if res[-1][0] - res[-2][0] > 1.15 * tipicos:
            res[-1] = (res[-2][0] + tipicos,) + tuple(res[-1][1:])
    return res, arr


def versos_manuais(nts, tempos, arr):
    """Inícios de verso informados pelo usuário -> mesmo formato de achar_versos."""
    pts = []
    for t in tempos:
        k = min(range(len(nts)), key=lambda q: abs(nts[q][0] - t))
        pts.append((nts[k][0], 0.0, k, arr.get(k)))
    pts.append((nts[-1][1], 0.0, len(nts), None))
    return pts


def acordes_dos_versos(nts, pts, n_partes, tonica, modo, vocab, troca=0.08, partilhar=0.8):
    """Divide cada verso em n_partes iguais e escolhe 1 acorde por parte.
    - as notas do arranque do verso seguinte contam com peso menor no último trecho;
    - versos com melodia parecida 'emprestam' notas uns aos outros, para que
      recebam a mesma harmonia (como um músico faria)."""
    versos = []
    for v in range(len(pts) - 1):
        t0, t1 = pts[v][0], pts[v + 1][0]
        k_ini, k_fim = pts[v][2], pts[v + 1][2]
        excl = set()
        info_prox = pts[v + 1][3]
        if info_prox:
            excl = set(range(info_prox[2], k_fim))
        bordas = np.linspace(t0, t1, n_partes + 1)
        perfis = np.zeros((n_partes, 12))
        melod = [[] for _ in range(n_partes)]
        grade = np.full(n_partes * 8, -1)
        for q in range(k_ini, k_fim):
            s_, e_, m = nts[q]
            # notas do arranque do verso seguinte: entram no último trecho com peso
            # menor; é ali que o acompanhamento costuma 'puxar' para o próximo verso
            peso_q = 0.7 if q in excl else 1.0
            for j in range(n_partes):
                a, b = bordas[j], bordas[j + 1]
                sobre = min(e_, b) - max(s_, a)
                if sobre <= 0:
                    continue
                peso = (1.3 if a <= s_ < a + 0.25 * (b - a) else 1.0) * peso_q
                perfis[j, m % 12] += sobre * peso
                if not melod[j] or melod[j][-1] != m % 12:
                    melod[j].append(m % 12)
            if q in excl:
                continue
            g0 = int((s_ - t0) / (t1 - t0) * len(grade))
            g1 = int(np.ceil((e_ - t0) / (t1 - t0) * len(grade)))
            grade[max(0, g0):min(len(grade), g1)] = m % 12
        versos.append({"ini": t0, "fim": t1, "bordas": bordas, "perfis": perfis,
                       "melodia": melod, "grade": grade, "arranque": pts[v][3],
                       "nota_ini": k_ini})
    # semelhança melódica entre versos
    nv = len(versos)
    sim = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(nv):
            if i != j:
                u, w = versos[i]["grade"], versos[j]["grade"]
                ok = (u >= 0) & (w >= 0)
                sim[i, j] = ((u == w) & ok).sum() / max(ok.sum(), 1) if ok.sum() > 8 else 0
    for i, v in enumerate(versos):
        comb = v["perfis"].copy()
        for j in range(nv):
            if sim[i, j] > 0.4:
                comb += partilhar * sim[i, j] * versos[j]["perfis"]
        v["comb"] = comb
        v["parecidos"] = [j for j in range(nv) if sim[i, j] > 0.4]
    # Viterbi sobre todas as partes, em sequência
    perfis = [p for v in versos for p in v["comb"]]
    acordes, conf, vazio, _ = escolher_acordes(
        [p / p.sum() if p.sum() > 0 else np.zeros(12) for p in perfis], vocab, tonica, modo, troca)
    q = 0
    for v in versos:
        v["acordes"], v["certeza"] = [], []
        for j in range(n_partes):
            v["acordes"].append(acordes[q])
            v["certeza"].append(None if vazio[q] else conf[q])
            q += 1
    # parte sem nota nenhuma herda o acorde anterior
    ant = None
    for v in versos:
        for j in range(n_partes):
            if v["certeza"][j] is None and ant is not None:
                v["acordes"][j] = ant
            ant = v["acordes"][j]
    return versos


# ----------------------------------------------------------------------------
# 7. Saídas
# ----------------------------------------------------------------------------
def notas_bracsa(acorde):
    """Notas do acorde a partir da fundamental (o músico escolhe a posição)."""
    return " ".join(acorde.grafia())


def voicing_midi(acorde, baixo_lim=55, alto_lim=71):
    """Posição fechada na região do brácsa (Sol3 a Si4) para o MIDI/prévia."""
    notas = []
    for pc in acorde.notas:
        m = baixo_lim + ((pc - baixo_lim) % 12)
        notas.append(m)
    notas.sort()
    return [m if m <= alto_lim else m - 12 for m in notas]


def nota_bogo(raiz):
    return 36 + ((raiz - 36) % 12)  # entre Dó2 e Si2


def fmt_tempo(s):
    return f"{int(s // 60)}:{s % 60:04.1f}"


def rotulo_certeza(c):
    if c is None:
        return "(pausa)"
    return "alta" if c > 0.08 else ("média" if c > 0.02 else "baixa - confira")


def escrever_cifra(caminho, linhas_ac, info, acordes_por_compasso=1):
    """linhas_ac: lista de dicts com ini, fim, acorde, certeza, melodia (tempos já
    na escala da gravação inteira)."""
    L = []
    L.append("ACOMPANHAMENTO SUGERIDO - brácsa e bőgő")
    L.append(f"Arquivo: {info['arquivo']}" + (f"   (trecho {info['trecho']})" if info.get("trecho") else ""))
    L.append(f"Tonalidade estimada: {NOMES_PT[info['tonica']]} {info['modo']}  "
             f"({NOMES[info['tonica']]}{'' if info['modo'] == 'maior' else 'm'})"
             f"   |   afinação do violino: Lá = {info['ref']:.0f} Hz")
    if info["modo_analise"] == "compasso":
        L.append(f"Modo: por compasso  |  andamento {info['bpm']:.0f} BPM, "
                 f"1 acorde a cada {info['tpa']} tempo(s)")
    else:
        L.append("Modo: livre (andamento irregular: os acordes seguem as mudanças da melodia, "
                 "com os tempos em que entram)")
    L.append("")

    if info["modo_analise"] == "compasso":
        compassos = []
        for i in range(0, len(linhas_ac), acordes_por_compasso):
            bloco = [r["acorde"].nome for r in linhas_ac[i:i + acordes_por_compasso]]
            nomes = [n if j == 0 or n != bloco[j - 1] else "-" for j, n in enumerate(bloco)]
            compassos.append(" ".join(nomes))
        L.append("CIFRA (4 compassos por linha)")
        for i in range(0, len(compassos), 4):
            grupo = compassos[i:i + 4]
            L.append(f"{i + 1:>4} | " + " | ".join(f"{c:<8}" for c in grupo) + " |")
    else:
        L.append("CIFRA (sequência de acordes; o tempo indica onde a linha começa)")
        for i in range(0, len(linhas_ac), 6):
            grupo = linhas_ac[i:i + 6]
            L.append(f"{fmt_tempo(grupo[0]['ini']):>7}  " +
                     "  ".join(f"{r['acorde'].nome:<5}" for r in grupo))
    L.append("")

    L.append("DETALHE (a última coluna mostra as notas que o violino toca durante cada acorde)")
    comp = info["modo_analise"] == "compasso"
    cab = (f"{'Comp.':>5}  " if comp else "") + \
        f"{'Tempo':>7}  {'Acorde':<6} {'Brácsa (notas)':<15} {'Bőgő':<5} {'Certeza':<16} Violino"
    L.append(cab)
    for i, r in enumerate(linhas_ac):
        a = r["acorde"]
        pre = f"{i // acordes_por_compasso + 1:>5}  " if comp else ""
        L.append(pre + f"{fmt_tempo(r['ini']):>7}  {a.nome:<6} {notas_bracsa(a):<15} "
                 f"{NOMES[a.raiz]:<5} {rotulo_certeza(r['certeza']):<16} {r['melodia']}")
    L.append("")
    L.append("Observações:")
    L.append("- 'Brácsa (notas)' lista as notas do acorde; a posição/digitação fica a seu critério.")
    L.append("- 'Bőgő' é a fundamental do acorde. Para alternar, a quinta é a 3ª nota da lista do brácsa.")
    L.append("- Linhas com certeza 'baixa' são as que mais vale conferir de ouvido.")
    texto = "\n".join(L)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    return texto


def escrever_cifra_versos(caminho, versos, nts, info, n_partes):
    def nome(m):
        return NOMES[m % 12]
    L = []
    L.append("ACOMPANHAMENTO SUGERIDO - brácsa e bőgő")
    L.append(f"Arquivo: {info['arquivo']}" + (f"   (trecho {info['trecho']})" if info.get("trecho") else ""))
    L.append(f"Tonalidade estimada: {NOMES_PT[info['tonica']]} {info['modo']}  "
             f"({NOMES[info['tonica']]}{'' if info['modo'] == 'maior' else 'm'})"
             f"   |   afinação do violino: Lá = {info['ref']:.0f} Hz")
    L.append(f"Modo: por verso  |  {len(versos)} versos, {n_partes} acordes por verso "
             f"(cada verso dividido em {n_partes} partes iguais)")
    if info.get("versos_manuais"):
        L.append("Inícios de verso informados manualmente.")
    L.append("")
    L.append("CIFRA (uma linha por verso)")
    larg = max(4, max(len(a.nome) for v in versos for a in v["acordes"]) + 1)
    for i, v in enumerate(versos):
        nomes = []
        for j, a in enumerate(v["acordes"]):
            nomes.append(a.nome if j == 0 or a.nome != v["acordes"][j - 1].nome else "-")
        L.append(f"V{i + 1:<2} {fmt_tempo(v['ini']):>7} | " + " | ".join(f"{x:<{larg}}" for x in nomes) + " |")
    L.append("('-' = repete o acorde anterior)")
    L.append("")
    L.append("VERSOS EM DETALHE")
    for i, v in enumerate(versos):
        arr = v["arranque"]
        entrada = ""
        if arr:
            notas_arr = " ".join(nome(nts[q][2]) for q in range(arr[2], v["nota_ini"] + 1))
            entrada = f"   entra com {arr[0]}: {notas_arr[:notas_arr.rfind(' ')]} → {notas_arr.split()[-1]}"
        parec = ""
        if v["parecidos"]:
            parec = "   (melodia parecida com " + ", ".join(f"V{j + 1}" for j in v["parecidos"]) + ")"
        L.append(f"Verso {i + 1}: {fmt_tempo(v['ini'])} a {fmt_tempo(v['fim'])}  "
                 f"({v['fim'] - v['ini']:.1f} s){entrada}{parec}")
        L.append(f"   {'Parte':>5}  {'Tempo':>7}  {'Acorde':<6} {'Brácsa (notas)':<15} {'Bőgő':<5} "
                 f"{'Certeza':<16} Violino")
        for j, a in enumerate(v["acordes"]):
            mel = " ".join(NOMES[pc] for pc in v["melodia"][j][:10])
            L.append(f"   {j + 1:>5}  {fmt_tempo(v['bordas'][j]):>7}  {a.nome:<6} {notas_bracsa(a):<15} "
                     f"{NOMES[a.raiz]:<5} {rotulo_certeza(v['certeza'][j]):<16} {mel}")
        L.append("")
    L.append("Observações:")
    L.append("- O verso começa na nota de chegada do felfutás/lefutás. As notas do arranque")
    L.append("  ficam no último trecho do verso anterior, com peso menor: ali o acompanhamento")
    L.append("  costuma 'puxar' para o 1º acorde do verso seguinte (ex.: D antes de G).")
    L.append("- Versos com melodia parecida recebem harmonia parecida de propósito.")
    L.append("- Se algum verso começou no lugar errado, informe os inícios com --versos")
    L.append("  (ex.: --versos 0:14.5,0:25.6,0:34.5) ou a duração com --duracao-verso, e rode de novo.")
    texto = "\n".join(L)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    return texto


def escrever_midi(caminho, eventos):
    """eventos: lista (canal, nota, inicio_s, fim_s, vel). MIDI tipo 0, 960 ticks/s."""
    tpq = 480  # tempo fixo 120 BPM -> 960 ticks por segundo, alinha com a gravação
    msgs = []
    for canal, nota, ini, fim, vel in eventos:
        msgs.append((int(ini * 960), 0x90 | canal, nota, vel))
        msgs.append((int(fim * 960), 0x80 | canal, nota, 0))
    msgs.sort(key=lambda m: (m[0], m[1] & 0xF0 == 0x90))
    def vlq(n):
        b = [n & 0x7F]
        n >>= 7
        while n:
            b.append((n & 0x7F) | 0x80)
            n >>= 7
        return bytes(b[::-1])
    trilha = bytearray()
    trilha += vlq(0) + b"\xFF\x51\x03" + (500000).to_bytes(3, "big")  # 120 BPM
    trilha += vlq(0) + bytes([0xC0, 32])   # canal 0: baixo acústico (bőgő)
    trilha += vlq(0) + bytes([0xC1, 41])   # canal 1: viola (brácsa)
    t_ant = 0
    for t, st, nota, vel in msgs:
        trilha += vlq(t - t_ant) + bytes([st, nota, vel])
        t_ant = t
    trilha += vlq(0) + b"\xFF\x2F\x00"
    with open(caminho, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq))
        f.write(b"MTrk" + struct.pack(">I", len(trilha)) + trilha)


def sintetizar(eventos, duracao):
    n = int(duracao * SR) + SR
    out = np.zeros(n, dtype=np.float32)
    for canal, nota, ini, fim, vel in eventos:
        f = 440 * 2 ** ((nota - 69) / 12)
        a, b = int(ini * SR), int(fim * SR)
        t = np.arange(b - a) / SR
        if canal == 0:  # bőgő: pizz/arco curto, grave
            onda = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)
            env = np.exp(-t * 3.0)
        else:  # brácsa: dente de serra suave
            onda = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 6))
            env = np.minimum(1, t / 0.03) * np.exp(-t * 1.2)
        env[-min(len(env), 400):] *= np.linspace(1, 0, min(len(env), 400))
        out[a:b] += (vel / 127) * 0.18 * onda * env
    return out


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def gerar_saidas_versos(args, base, y, versos, nts, desloc, trecho, tonica, modo, ref):
    duracao = len(y) / SR
    eventos = []
    for v in versos:
        b = v["bordas"]
        for j, a in enumerate(v["acordes"]):
            ini, fim = b[j], b[j + 1]
            mudou = j == 0 or a.nome != v["acordes"][j - 1].nome
            eventos.append((0, nota_bogo(a.raiz), ini, min(fim, ini + 0.8), 100 if mudou else 85))
            for m in voicing_midi(a):
                eventos.append((1, m, ini, fim - 0.02, 72 if mudou else 60))
    escrever_midi(base + "_acomp.mid", eventos)
    if not args.sem_previa:
        acomp = sintetizar(eventos, duracao)
        mix = np.zeros(max(len(acomp), len(y)), dtype=np.float32)
        mix[:len(y)] += 0.8 * y
        mix[:len(acomp)] += acomp
        mix /= max(1e-6, np.abs(mix).max()) / 0.95
        wavfile.write(base + "_previa.wav", SR, (mix * 32767).astype(np.int16))
    # tempos da gravação inteira na cifra
    for v in versos:
        v["ini"] += desloc
        v["fim"] += desloc
        v["bordas"] = v["bordas"] + desloc
    nts_abs = [[s_ + desloc, e_ + desloc, m] for s_, e_, m in nts]
    info = {"arquivo": os.path.basename(args.audio), "trecho": trecho, "tonica": tonica,
            "modo": modo, "ref": ref, "versos_manuais": bool(args.versos)}
    texto = escrever_cifra_versos(base + "_acordes.txt", versos, nts_abs, info, args.acordes_por_verso)
    print()
    print(texto)
    print()
    print("Arquivos gerados:")
    for suf in ("_acordes.txt", "_acomp.mid") + (() if args.sem_previa else ("_previa.wav",)):
        print("  " + base + suf)


def ler_segundos(txt):
    """'75', '75.5' ou '1:15' -> segundos."""
    if txt is None:
        return None
    if ":" in txt:
        m, s_ = txt.split(":")
        return int(m) * 60 + float(s_)
    return float(txt)


def main():
    ap = argparse.ArgumentParser(description="Sugere acordes de brácsa e bőgő para uma gravação de violino.")
    ap.add_argument("audio")
    ap.add_argument("--de", help="analisa a partir deste ponto (ex.: 1:10 ou 70)")
    ap.add_argument("--ate", help="analisa até este ponto (ex.: 1:10 ou 70)")
    ap.add_argument("--modo", choices=["auto", "verso", "compasso", "livre"], default="auto",
                    help="verso: acha os versos (felfutás/lefutás, repetição) e divide cada um "
                         "em partes iguais; compasso: 1 acorde a cada N tempos (pulso regular); "
                         "livre: acordes seguem a melodia; auto (padrão): verso quando acha "
                         "versos, senão compasso ou livre")
    ap.add_argument("--acordes-por-verso", type=int, default=8,
                    help="modo verso: em quantas partes iguais cada verso é dividido (padrão 8)")
    ap.add_argument("--duracao-verso", type=float,
                    help="modo verso: duração aproximada de um verso em segundos, se a "
                         "detecção pegar o dobro ou a metade")
    ap.add_argument("--versos",
                    help="modo verso: inícios dos versos, separados por vírgula "
                         "(ex.: 0:14.5,0:25.6,0:34.5); se omitido, detecta")
    ap.add_argument("--bpm", type=float, help="andamento em batidas por minuto (se omitido, detecta)")
    ap.add_argument("--inicio", type=float, help="segundo do primeiro tempo (usar junto com --bpm)")
    ap.add_argument("--tempos-por-acorde", type=int, default=2)
    ap.add_argument("--tempos-por-compasso", type=int, default=2)
    ap.add_argument("--dur-min", type=float, default=1.0,
                    help="modo livre: duração mínima de cada acorde em segundos (padrão 1.0)")
    ap.add_argument("--estabilidade", type=float,
                    help="penalidade por trocar de acorde; maior = menos trocas "
                         "(padrão 0.1 no modo compasso, 0.6 no livre)")
    ap.add_argument("--sem-setima", action="store_true", help="escrever só tríades (A em vez de A7)")
    ap.add_argument("--tonalidade", help="força a tonalidade, ex.: D, Am, G")
    ap.add_argument("--padrao", choices=["sustentado", "contra"], default="sustentado",
                    help="ritmo do acompanhamento na prévia/MIDI no modo compasso (a cifra não muda)")
    ap.add_argument("--sem-previa", action="store_true")
    args = ap.parse_args()

    base = os.path.splitext(args.audio)[0]
    print("Lendo áudio...")
    y = carregar_audio(args.audio)
    de, ate = ler_segundos(args.de), ler_segundos(args.ate)
    desloc = 0.0
    trecho = None
    if de is not None or ate is not None:
        desloc = de or 0.0
        fim_s = ate if ate is not None else len(y) / SR
        y = y[int(desloc * SR):int(fim_s * SR)]
        trecho = f"{fmt_tempo(desloc)} a {fmt_tempo(fim_s)}"
        base += f"_{int(desloc)}s-{int(fim_s)}s"
        if len(y) < SR:
            sys.exit("Trecho curto demais.")
    duracao = len(y) / SR

    ref = estimar_afinacao(y)
    S = stft_mag(y)
    notas, sal = saliencia_notas(S, ref=ref)
    croma = croma_de_notas(notas, sal)
    tempos_quadro = np.arange(S.shape[0]) * HOP / SR

    # tonalidade
    if args.tonalidade:
        t = args.tonalidade.strip()
        modo = "menor" if t.endswith("m") else "maior"
        tonica = nome_para_classe(t[:-1] if t.endswith("m") else t)
    else:
        tonica, modo = estimar_tonalidade(croma.sum(axis=0), croma)
    definir_grafia(tonica, modo)
    # analisa sempre com sétimas (a melodia toca a sétima da dominante com frequência);
    # com --sem-setima, só simplifica para tríade na hora de escrever
    vocab = vocabulario(True)

    modo_analise = args.modo
    if args.versos and modo_analise == "auto":
        modo_analise = "verso"
    if modo_analise in ("auto", "verso"):
        print("Transcrevendo a melodia e procurando os versos...")
        nts = transcrever(y, ref)
        if len(nts) < 12:
            if modo_analise == "verso":
                sys.exit("Poucas notas na gravação para achar versos.")
        else:
            arr = arranques(nts)
            if args.versos:
                tempos = [ler_segundos(t) - desloc for t in args.versos.split(",")]
                pts = versos_manuais(nts, tempos, arr)
            else:
                L, forca_rep = periodo_repeticao(nts, duracao)
                if args.duracao_verso:
                    L, forca_rep = args.duracao_verso, max(forca_rep, 0.15)
                pts = achar_versos(nts, duracao, L)[0] if L else []
                n_arr = sum(1 for p_ in pts[1:-1] if p_[3])
                confiavel = len(pts) >= 3 and (n_arr >= 1 or forca_rep >= 0.15)
                if not confiavel and modo_analise == "auto":
                    pts = []
                elif not confiavel:
                    print("  aviso: não achei versos com clareza; o resultado pode estar impreciso")
            if len(pts) >= 2:
                modo_analise = "verso"
                troca = 0.08 if args.estabilidade is None else args.estabilidade
                versos = acordes_dos_versos(nts, pts, args.acordes_por_verso, tonica, modo,
                                            vocab, troca=troca)
                if args.sem_setima:
                    for v in versos:
                        v["acordes"] = [Acorde(a.raiz, "") if a.tipo == "7" else a for a in v["acordes"]]
                gerar_saidas_versos(args, base, y, versos, nts, desloc, trecho, tonica, modo, ref)
                return
        if modo_analise == "verso":
            sys.exit("Não consegui achar versos. Informe os inícios com --versos.")
        modo_analise = "auto"

    print("Detectando o pulso...")
    S_curto = stft_mag(y, 1024)
    env = curva_de_ataques(S_curto, sal)
    bpm = args.bpm or estimar_bpm(env)
    if modo_analise == "auto":
        if args.bpm or pulso_estavel(env, bpm):
            modo_analise = "compasso"
        else:
            modo_analise = "livre"
            print("  andamento irregular nesta gravação: usando o modo livre "
                  "(use --modo compasso --bpm N para forçar compassos)")

    print("Escolhendo acordes...")
    linhas_ac = []
    bpm_real = None
    tpa = args.tempos_por_acorde
    if modo_analise == "livre":
        troca = 0.6 if args.estabilidade is None else args.estabilidade
        for a, b, ac, cert in segmentar_livre(croma, tonica, modo, vocab, troca=troca,
                                              dur_min=args.dur_min):
            linhas_ac.append({"ini": a, "fim": b, "acorde": ac, "certeza": cert})
    else:
        estab = 0.1 if args.estabilidade is None else args.estabilidade
        if args.bpm and args.inicio is not None:
            batidas = grade_fixa(bpm, args.inicio - desloc, duracao)
        elif args.bpm:
            batidas = rastrear_batidas(env, bpm, rigidez=400)
        else:
            batidas = rastrear_batidas(env, bpm)
            # segunda passada com o andamento medido nas próprias batidas
            if len(batidas) > 8:
                meio = batidas[len(batidas) // 8: -len(batidas) // 8 or None]
                bpm = 60 / np.median(np.diff(meio))
                batidas = rastrear_batidas(env, bpm)
        # descarta batidas antes do violino começar a soar e completa as que faltarem
        rms = np.sqrt((S_curto ** 2).sum(axis=1))
        t_curto = np.arange(len(rms)) * HOP / SR
        com_som = t_curto[rms > 0.1 * np.percentile(rms, 95)]
        if len(com_som) and args.inicio is None and len(batidas):
            primeiro = com_som[0]
            periodo = 60 / bpm
            while batidas[0] - periodo >= primeiro - 0.06:
                batidas = np.concatenate([[batidas[0] - periodo], batidas])
            batidas = batidas[batidas >= primeiro - 0.06]
        if len(batidas) < 4:
            sys.exit("Não consegui achar o pulso. Tente --modo livre ou informe --bpm.")
        bpm_real = 60 / np.mean(np.diff(batidas))

        def analisar(batidas_fase):
            marcos = list(batidas_fase[::tpa])
            marcos.append(min(duracao, marcos[-1] + tpa * 60 / bpm_real))
            perfis, energias = [], []
            for a, b in zip(marcos[:-1], marcos[1:]):
                sel = (tempos_quadro >= a) & (tempos_quadro < b)
                bloco = croma[sel]
                if len(bloco) == 0:
                    perfis.append(np.zeros(12))
                    energias.append(0.0)
                    continue
                # o início do segmento (tempo forte) pesa mais
                peso = np.linspace(1.5, 0.8, len(bloco))[:, None]
                p = (bloco * peso).sum(axis=0)
                energias.append(float(bloco.sum() / len(bloco)))
                perfis.append(p / p.sum() if p.sum() > 0 else np.zeros(12))
            # trechos quase sem som (pausas, fim da gravação) não recebem acorde próprio
            ref_e = np.median([e for e in energias if e > 0] or [1])
            perfis = [p if e > 0.25 * ref_e else np.zeros(12) for p, e in zip(perfis, energias)]
            while len(perfis) > 1 and perfis[-1].sum() == 0:
                perfis.pop()
                marcos.pop()
            acordes, conf, vazio, _ = escolher_acordes(perfis, vocab, tonica, modo, estab)
            # coerência: quão bem o melhor acorde de cada trecho explica as notas
            cheios = [p for p in perfis if p.sum() > 0]
            coer = np.mean([max(pontuar(a, p) + bonus_tonal(a, tonica, modo) for a in vocab)
                            for p in cheios]) if cheios else -1
            return marcos, acordes, conf, vazio, coer

        if args.inicio is not None:
            marcos, acordes, conf, vazio, _ = analisar(batidas)
        else:
            # testa cada batida como 'tempo forte' e também a grade deslocada em
            # meio tempo (caso o pulso tenha travado no contratempo)
            meio = 0.5 * np.median(np.diff(batidas))

            def acento(g):
                i = np.clip(np.round(g * SR / HOP).astype(int), 1, len(env) - 2)
                return np.maximum(np.maximum(env[i - 1], env[i]), env[i + 1]).mean()
            # nota de cada grade = coerência harmônica + acentuação do arco
            grades = [batidas, batidas + meio]
            acentos = [acento(g) for g in grades]
            melhor_ac = max(acentos)
            candidatos = []
            for g, ac_g in zip(grades, acentos):
                for f in range(tpa):
                    r = analisar(g[f:])
                    nota = r[4] + 0.5 * np.log(max(ac_g, 1e-9) / max(melhor_ac, 1e-9))
                    candidatos.append((nota, r))
            marcos, acordes, conf, vazio, _ = max(candidatos, key=lambda c: c[0])[1]
        for i, ac in enumerate(acordes):
            linhas_ac.append({"ini": marcos[i], "fim": marcos[i + 1], "acorde": ac,
                              "certeza": None if vazio[i] else conf[i]})

    if not linhas_ac:
        sys.exit("Não encontrei notas na gravação.")
    for i, r in enumerate(linhas_ac):
        if args.sem_setima and r["acorde"].tipo == "7":
            r["acorde"] = Acorde(r["acorde"].raiz, "")
        if r["certeza"] is None and i > 0:  # pausa herda o acorde anterior
            r["acorde"] = linhas_ac[i - 1]["acorde"]
        r["melodia"] = melodia_trecho(notas, sal, r["ini"], r["fim"])

    # eventos para MIDI e prévia (tempos relativos ao trecho analisado)
    eventos = []
    for i, r in enumerate(linhas_ac):
        if r["certeza"] is None:
            continue
        a, ini, fim = r["acorde"], r["ini"], r["fim"]
        voz = voicing_midi(a)
        if modo_analise == "livre":
            # bőgő na entrada do acorde e de novo a cada ~1,2 s se ele for longo
            n_golpes = max(1, int((fim - ini) / 1.2))
            for k in range(n_golpes):
                t0 = ini + k * (fim - ini) / n_golpes
                eventos.append((0, nota_bogo(a.raiz), t0, min(fim, t0 + 0.8), 100 if k == 0 else 85))
            for m in voz:
                eventos.append((1, m, ini, fim - 0.02, 70))
            continue
        batida_dur = 60 / bpm_real
        eventos.append((0, nota_bogo(a.raiz), ini, ini + 0.9 * batida_dur, 100))
        if args.padrao == "sustentado":
            for m in voz:
                eventos.append((1, m, ini, fim - 0.02, 70))
        else:
            # contratempo: um golpe no meio de cada tempo
            for k in range(tpa):
                t0 = ini + (k + 0.5) * batida_dur
                if t0 < fim:
                    for m in voz:
                        eventos.append((1, m, t0, t0 + 0.4 * batida_dur, 75))
            if tpa > 1:
                # baixo também no 2º tempo, com a quinta
                t1 = ini + batida_dur
                eventos.append((0, nota_bogo(a.notas[2]), t1, t1 + 0.9 * batida_dur, 90))
    escrever_midi(base + "_acomp.mid", eventos)

    # cifra com os tempos da gravação inteira
    for r in linhas_ac:
        r["ini"] += desloc
        r["fim"] += desloc
    info = {"arquivo": os.path.basename(args.audio), "trecho": trecho, "tonica": tonica,
            "modo": modo, "bpm": bpm_real, "tpa": tpa, "ref": ref, "modo_analise": modo_analise}
    acordes_por_compasso = max(1, args.tempos_por_compasso // tpa)
    texto = escrever_cifra(base + "_acordes.txt", linhas_ac, info, acordes_por_compasso)

    if not args.sem_previa:
        acomp = sintetizar(eventos, duracao)
        mix = np.zeros(max(len(acomp), len(y)), dtype=np.float32)
        mix[:len(y)] += 0.8 * y
        mix[:len(acomp)] += acomp
        mix /= max(1e-6, np.abs(mix).max()) / 0.95
        wavfile.write(base + "_previa.wav", SR, (mix * 32767).astype(np.int16))

    print()
    print(texto)
    print()
    print("Arquivos gerados:")
    print("  " + base + "_acordes.txt")
    print("  " + base + "_acomp.mid")
    if not args.sem_previa:
        print("  " + base + "_previa.wav")


if __name__ == "__main__":
    main()
