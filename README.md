# Aplikacja do Rozpoznawania Emocji

Aplikacja webowa napisana w Pythonie z wykorzystaniem frameworka FastAPI umożliwia rozpoznawanie emocji na podstawie zdjęć oraz obrazu z kamery. Użytkownik może także przeprowadzić proces trenowania własnego modelu uczenia maszynowego na dostarczonych danych. Narzędzie jest dedykowane do analizy emocji oraz eksperymentowania z algorytmami rozpoznawania emocji.

---

# Spis treści

1. [Wymagania i Instalacja](#wymagania-i-instalacja)
    - [Wymagania wstępne](#wymagania-wstępne)
    - [Kroki instalacji](#kroki-instalacji)
2. [Uruchamianie aplikacji](#uruchamianie-aplikacji)
    - [Uruchomienie serwisu](#uruchomienie-serwisu)
    - [Uczenie własnego modelu](#uczenie-własnego-modelu)
3. [Opis struktury projektu](#opis-struktury-projektu)
4. [Opis testów aplikacji](#opis-testow-aplikacji)

---

## Wymagania i Instalacja

### Wymagania wstępne

- **Python**: 3.10.12
- **Conda** lub **venv** (dla środowisk wirtualnych)
    ```bash
    sudo apt-get install build-essential cmake
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libboost-all-dev
    ```

---

### Kroki instalacji

#### 1. Utworzenie środowiska wirtualnego:

- **Za pomocą Conda:**
  ```bash
  conda create --name myenv python=3.10.12
  conda activate myenv
  ```

- **Za pomocą venv:**
  ```bash
  python3.10 -m venv myenv
  source myenv/bin/activate  # Linux/Mac
  myenv\Scripts\activate     # Windows
  ```

#### 2. Instalacja wymaganych pakietów:
  ```bash
  pip install -r requirements.txt
  ```

#### 3. Pobranie repozytorium:
  ```bash
  git clone https://github.com/dwyszyns/recognizing_emotions.git
  ```

---

## Uruchamianie aplikacji

### Uruchomienie serwisu

Aby uruchomić serwis do rozpoznawania emocji ze zdjęć oraz obrazu z kamery, wykonaj następujące polecenie w terminalu:
  ```bash
  uvicorn main:app --reload
  ```

### Uczenie własnego modelu

1. Pobierz odpowiedni zbór danych i podziel go na dwa katalogi: `train` oraz `test`.
2. Uruchom jedną z metod znajdujących się w folderze `src` w celu nauczenia modelu.

#### Najwyższe uzyskane średnie wyniki dokładności - przy użyciu algorytmu z pliku `complex_cnn.py`:
- **RAF-DB**:
  - Testowe: 56%
  - Treningowe: 66%
- **CK+:**
  - Testowe: 81%
  - Treningowe: 90%

---

## Opis struktury projektu

Tu opisać wszystkie pliki i ich zawartości.

---

## Opis testów aplikacji

Tu opisać wszystkie pliki i ich zawartości.

