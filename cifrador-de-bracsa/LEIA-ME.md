# Cifrador de Brácsa

Programa para gravar ou abrir uma gravação de violino, marcar os versos e gerar o acompanhamento
de brácsa e bőgő (cifra + reprodução nos estilos ugrós, lassú csárdás dunántúli,
lassú füzesi, ritka füzesi, román forgatós e segurado).

## O que tem nesta pasta

| Arquivo / pasta | Para que serve |
|---|---|
| `Cifrador-de-Bracsa.html` | O programa inteiro em um arquivo só. Dois cliques e ele abre no navegador do PC. |
| `app/` | O mesmo programa como **aplicativo instalável** (celular e PC). É esta pasta que vai para a internet. |
| `Rodar-no-PC-Windows.bat` | Abre a pasta `app` como aplicativo no Windows, sem internet (precisa do Python). |
| `rodar-no-pc-mac-linux.sh` | O mesmo para Mac e Linux. |
| `script-python/acompanhamento.py` | O script de linha de comando da primeira versão (análise automática). |

---

## 1. No computador

### Jeito mais fácil (sem instalar nada)
Dê dois cliques em **`Cifrador-de-Bracsa.html`**. Ele abre no Chrome, Edge ou Firefox e
funciona sem internet. Já vem com a música de exemplo carregada; use **Abrir gravação**
para abrir as suas. As marcações ficam guardadas no navegador, e o botão **Salvar projeto**
gera um `.json` para você continuar depois (ou em outro computador).

### Como aplicativo instalado no PC (ícone próprio, janela própria)
1. Instale o Python: https://www.python.org/downloads/ (no Windows, marque
   **"Add Python to PATH"** durante a instalação).
2. Dê dois cliques em **`Rodar-no-PC-Windows.bat`** (no Mac/Linux: `rodar-no-pc-mac-linux.sh`).
   Abre uma janela preta (deixe aberta) e o navegador em `http://localhost:8765/`.
3. No Chrome ou Edge, clique no ícone **Instalar** na barra de endereço
   (um monitor com uma seta). Pronto: o Cifrador vira um app no menu Iniciar / Dock.

Se você publicar o app na internet (parte 2), dá para instalar no PC direto pelo endereço,
sem Python.

## 2. No celular (aplicativo instalável)

O celular precisa abrir o app de um endereço na internet uma vez; depois ele fica
instalado e funciona offline. O jeito gratuito mais simples é o **GitHub Pages**:

1. Crie uma conta em https://github.com (gratuita).
2. Clique em **New repository**, dê o nome `cifrador`, deixe **Public** e crie.
3. Na página do repositório, clique em **Add file → Upload files** e arraste
   **o conteúdo da pasta `app`** (os arquivos `index.html`, `sw.js`, `manifest.webmanifest`,
   `exemplo.m4a` e a pasta `icons`, não a pasta `app` em si). Clique em **Commit changes**.
4. Vá em **Settings → Pages**. Em *Branch*, escolha `main` e `/ (root)` e clique em **Save**.
5. Espere 1 ou 2 minutos. O endereço fica: `https://SEU-USUARIO.github.io/cifrador/`

**Atenção:** com o repositório público, qualquer pessoa com o endereço consegue abrir o app,
inclusive o áudio de exemplo, que é um trecho da sua gravação (`exemplo.m4a`). Se não quiser
isso, não envie esse arquivo: o app abre vazio e pede uma gravação.

### Instalar no celular
- **Android (Chrome):** abra o endereço → menu **⋮** → **Instalar app** (ou
  *Adicionar à tela inicial*).
- **iPhone (Safari):** abra o endereço → botão **Compartilhar** → **Adicionar à Tela de Início**.

O ícone vermelho aparece na tela inicial e o app abre em tela cheia, também sem internet.

### Usando no celular
- **Abrir gravação** abre o seletor de arquivos. No iPhone, salve antes a gravação do app
  Gravador em *Arquivos* (Compartilhar → Salvar em Arquivos).
- Para marcar versos e trocas de acorde enquanto ouve, use os botões **+ Verso aqui** e
  **+ Troca aqui**. Toque numa linha para selecioná-la e arraste para ajustar.
- A **Velocidade** deixa a gravação mais lenta sem mudar a afinação; o acompanhamento
  acompanha.

## 3. Gravar direto no programa

Clique em **Gravar** (no topo). Na primeira vez o navegador pede permissão para usar o
microfone: aceite.

1. Toque no **botão vermelho** e toque o violino. A barra mostra o nível do som: se aparecer
   *Muito alto*, afaste o aparelho; se aparecer *Muito baixo*, aproxime.
2. Toque de novo para parar (no PC, a barra de espaço também para). A gravação abre sozinha
   para análise: siga com os versos e os acordes normalmente.
3. Todas as gravações ficam guardadas **no próprio aparelho**, na lista da área Gravar.
   Em cada uma: **Analisar** abre de novo, **Salvar arquivo** baixa o áudio
   (.webm no Chrome/Android, .m4a no iPhone) e **Apagar** remove (toque duas vezes para confirmar).

Onde a gravação funciona:
- **App instalado** (celular ou PC) e o atalho **Rodar-no-PC**: sim.
- **`Cifrador-de-Bracsa.html` com dois cliques**: sim no Chrome, Edge e Firefox. Se o
  navegador recusar o microfone nesse modo, use o atalho Rodar-no-PC.
- **Link do Claude**: não. Lá a página não tem permissão de usar o microfone; grave no app
  ou no gravador do celular e use *Abrir gravação*.

O programa grava sem os filtros de voz do navegador (redução de ruído e controle automático
de volume), que deixariam o violino abafado e com volume oscilando.

## 4. Atualizar o app depois

Se eu (ou alguém) mandar uma versão nova:
1. Substitua os arquivos no GitHub (Add file → Upload files, com os novos por cima).
2. No `sw.js`, aumente o número em `cifrador-v2` (para `cifrador-v3`, e assim por diante). É isso que
   avisa os celulares que há versão nova. Eles atualizam na segunda vez que o app abrir.

## 5. Script Python (opcional)

O `script-python/acompanhamento.py` faz a análise automática por linha de comando,
gerando cifra em texto, MIDI e uma prévia em áudio.

```
pip install numpy scipy
python acompanhamento.py minha_gravacao.m4a --de 1:11
python acompanhamento.py minha_gravacao.m4a --versos 0:14.5,0:26.0,0:33.9 --acordes-por-verso 8
```
Para abrir m4a/mp3 é preciso ter o **ffmpeg** instalado (WAV funciona sem ele).
Veja todas as opções com `python acompanhamento.py --help`.

## Dicas e limites

- O programa funciona melhor com **violino solo**. Com a banda inteira tocando, o brácsa
  atrapalha a leitura das notas.
- O áudio é analisado no próprio aparelho; nada é enviado para a internet.
- As marcações ficam guardadas no navegador/aparelho onde você as fez. Para levar a outro
  aparelho, use **Salvar projeto** e depois **Abrir projeto** (abra a gravação primeiro).
