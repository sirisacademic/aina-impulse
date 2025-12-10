# IMPULS - Manual d'Usuari

## Guia completa per a usuaris del sistema de cerca semàntica multilingüe

---

## 1. Introducció

### Què és IMPULS?

IMPULS és un sistema de cerca semàntica intel·ligent que permet trobar projectes d'R+D utilitzant llenguatge natural en **català, castellà o anglès**. A diferència dels cercadors tradicionals basats en paraules clau exactes, IMPULS entén el significat de la teva consulta i troba projectes rellevants encara que no continguin exactament les mateixes paraules.

### Característiques principals

| Característica | Descripció |
|----------------|------------|
| **Cerca multilingüe** | Cerca en català, castellà o anglès i troba resultats en qualsevol idioma |
| **Comprensió intel·ligent** | Interpreta consultes complexes amb filtres i preferències |
| **Expansió de conceptes** | Inclou sinònims i termes relacionats automàticament |
| **Filtres estructurats** | Refina per programa de finançament, any, ubicació, etc. |

### Accés al sistema

- **Interfície web**: http://impuls-aina.sirisacademic.com:8080/impuls_ui.html
- **API REST**: http://impuls-aina.sirisacademic.com:8000
- **Documentació API**: http://impuls-aina.sirisacademic.com:8000/docs

---

## 2. Interfície Web

### 2.1 Pantalla principal

En accedir a la interfície web trobaràs:

```
┌─────────────────────────────────────────────────────────────┐
│  IMPULS - Sistema de Cerca Semàntica                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [═══════════════════════════════════] [Cercar]             │
│   Introdueix la teva consulta...                            │
│                                                             │
│  ☐ Parsing intel·ligent    ☐ Expansió de consulta           │
│                                                             │
│  [▼ Filtres de Metadades]                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Realitzar una cerca bàsica

1. **Escriu la teva consulta** al camp de cerca
2. Fes clic a **"Cercar"** o prem Enter
3. Els resultats apareixeran ordenats per rellevància

**Exemples de consultes simples:**
- `intel·ligència artificial`
- `renewable energy`
- `biotecnología marina`

### 2.3 Cerca amb parsing intel·ligent

Activa l'opció **"Parsing intel·ligent"** perquè el sistema interpreti consultes complexes automàticament.

**Exemples de consultes complexes:**

| Consulta | El sistema entén |
|----------|------------------|
| `projectes d'IA en salut finançats per H2020 des de 2020` | Tema: IA + salut, Programa: H2020, Any: ≥2020 |
| `machine learning research at Catalan universities` | Tema: machine learning, Org: universitat, Ubicació: Catalunya |
| `proyectos de energía renovable en colaboración con empresas` | Tema: energia renovable, Org: empresa |

Quan el parsing està actiu, veuràs un panell de **feedback** que mostra com el sistema ha interpretat la teva consulta.

### 2.4 Filtres de metadades

Expandeix el panell **"Filtres de Metadades"** per refinar la teva cerca:

| Filtre | Descripció | Exemple |
|--------|------------|---------|
| **Framework** | Programa de finançament | H2020, Horizon Europe, FEDER, SIFECAT |
| **RIS3CAT Àmbit** | Àmbit sectorial | Salut, Energia, TIC, Indústria |
| **Any des de** | Any mínim | 2018 |
| **Any fins a** | Any màxim | 2024 |

### 2.5 Expansió de consulta

Activa **"Expansió de consulta"** perquè el sistema afegeixi automàticament:
- **Sinònims**: "IA" → "intel·ligència artificial", "inteligencia artificial", "artificial intelligence"
- **Traduccions**: termes equivalents en altres idiomes
- **Conceptes relacionats**: termes més amplis o específics

Quan l'expansió està activa, veuràs els termes afegits al panell de resultats.

---

## 3. Exemples d'ús per escenari

### 3.1 Investigador cercant projectes en la seva àrea

**Objectiu**: Trobar projectes de blockchain al sector financer

```
Consulta: "blockchain aplicat a finances i banca"
Opcions: ☑ Parsing intel·ligent  ☑ Expansió
```

**Resultat**: El sistema trobarà projectes sobre:
- Blockchain, cadena de blocs, distributed ledger
- Fintech, serveis financers, banca digital

### 3.2 Gestor de polítiques d'R+D

**Objectiu**: Veure quins projectes H2020 sobre economia circular hi ha a Catalunya des de 2019

```
Consulta: "projectes H2020 economia circular Catalunya 2019"
Opcions: ☑ Parsing intel·ligent
```

**El sistema extreu automàticament**:
- Programa: Horizon 2020
- Tema: economia circular
- Ubicació: Catalunya
- Any: ≥2019

### 3.3 Empresa cercant col·laboradors

**Objectiu**: Trobar projectes de robòtica amb participació de centres de recerca

```
Consulta: "robòtica industrial amb centres de recerca"
Opcions: ☑ Parsing intel·ligent
Filtre manual: Organization type = REC (Research Center)
```

### 3.4 Cerca multilingüe

**Objectiu**: Trobar tots els projectes sobre un tema independentment de l'idioma

```
Consulta en català: "aprenentatge automàtic per diagnòstic mèdic"
```

El sistema trobarà projectes que mencionen:
- "aprenentatge automàtic" (CA)
- "machine learning" (EN)
- "aprendizaje automático" (ES)

---

## 4. Interpretació de resultats

### 4.1 Informació de cada resultat

Cada resultat mostra:

```
┌─────────────────────────────────────────────────────────────┐
│ #1 | Score: 0.8745                                          │
├─────────────────────────────────────────────────────────────┤
│ Títol del projecte                                          │
│ ID: PROJECT_12345                                           │
├─────────────────────────────────────────────────────────────┤
│ Resum del projecte que descriu els objectius i              │
│ metodologia de la recerca...                                │
├─────────────────────────────────────────────────────────────┤
│ Framework: Horizon 2020  │ Àmbit: TIC, Bio  │ Any: 2021     │
│ Matched by: query, machine learning, IA                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Puntuació de rellevància (Score)

- **0.90 - 1.00**: Molt alta rellevància, coincidència gairebé exacta
- **0.75 - 0.90**: Alta rellevància, molt relacionat amb la consulta
- **0.60 - 0.75**: Rellevància mitjana, relacionat però no exacte
- **< 0.60**: Baixa rellevància, relació tangencial

### 4.3 Camp "Matched by"

Indica quins termes han provocat que el document aparegui als resultats:
- `query`: Ha coincidit amb la teva consulta original
- Altres termes: Ha coincidit amb termes d'expansió

---

## 5. Consells per a millors resultats

### ✅ Bones pràctiques

1. **Usa llenguatge natural**: Escriu com si preguntessis a una persona
   - ✅ `projectes sobre IA en hospitals catalans`
   - ❌ `IA AND hospital AND Catalunya`

2. **Sigues específic quan calgui**: Afegeix context
   - ✅ `energia solar fotovoltaica per a edificis`
   - ❌ `solar`

3. **Usa l'idioma que prefereixis**: El sistema entén CA/ES/EN
   - ✅ Qualsevol idioma funciona igual de bé

4. **Combina cerca semàntica amb filtres**: Per a més precisió
   - ✅ Consulta: `machine learning` + Filtre: Framework=H2020, Any≥2020

### ❌ Evitar

1. **Consultes massa curtes**: Una sola paraula dona resultats molt amplis
2. **Operadors booleans**: AND, OR, NOT no són necessaris
3. **Caràcters especials**: Evita `"`, `*`, `?` a la consulta

---

## 6. Preguntes freqüents (FAQ)

### Per què no trobo un projecte específic?

- Verifica que el projecte estigui a la base de dades (actualment ~27.000 projectes de RIS3CAT)
- Prova amb diferents termes o sinònims
- Desactiva els filtres per a una cerca més àmplia

### Quins idiomes puc usar?

Pots cercar en **català, castellà o anglès**. El sistema trobarà resultats rellevants independentment de l'idioma del document.

### Què significa cada programa de finançament?

| Codi | Programa |
|------|----------|
| H2020 | Horizon 2020 (UE, 2014-2020) |
| HE | Horizon Europe (UE, 2021-2027) |
| FEDER | Fons Europeu de Desenvolupament Regional |
| SIFECAT | Sistema de finançament català |
| AEI | Agencia Estatal de Investigación (Espanya) |

### Com funciona l'expansió de consulta?

El sistema utilitza una base de coneixement de 4.265 conceptes d'R+D amb:
- Etiquetes en múltiples idiomes
- Sinònims i àlies
- Relacions jeràrquiques (conceptes més amplis/específics)

### Puc accedir a l'API programàticament?

Sí, l'API REST està disponible a `http://impuls-aina.sirisacademic.com:8000`. Consulta la documentació a `/docs` per a més detalls.

---

## 7. Suport i contacte

### Reportar problemes

Si trobes errors o tens suggeriments:
- **GitHub Issues**: https://github.com/sirisacademic/aina-impulse/issues

### Més informació

- **Projecte AINA**: https://projecteaina.cat/
- **SIRIS Academic**: https://sirisacademic.com/
- **Documentació tècnica**: Consulteu `/docs` al repositori

---

*IMPULS - Projecte AINA Challenge 2024*
*Desenvolupat per SIRIS Academic per a la Generalitat de Catalunya*
