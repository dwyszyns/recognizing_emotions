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
  - dane testowe: 56%
  - dane treningowe: 66%
- **CK+:**
  - dane testowe: 81%
  - dane treningowe: 90%

---

## Opis struktury projektu

- **src**  
  Folder zawierający kody modeli uczenia maszynowego używane do trenowania:
  - `adaboost_algorithm.py` – implementacja modelu AdaBoost.
  - `random_forest_algorithm.py` – implementacja modelu Random Forest.
  - `complex_cnn.py` – implementacja złożonych sieci konwolucyjnych przy użyciu biblioteki Tensorflow.
  - `simple_cnn.py` – implementacja uproszczonej wersji sieci konwolucyjnych napisanych od zera.
  - `functions.py` – implementacja funkcji, które są wykorzystywane podczas uczenia wymienionych metod. Znajdują się tam funkcje, które służą do przetwarzania obrazu ze zbioru danych oraz ekstrakcji cech ze zdjęć.

- **static**  
  Zawiera zasoby statyczne wykorzystywane w aplikacji:
  - Folder `css`: Plik `style.css`, który definiuje styl aplikacji.
  - Zdjęcia: Obrazy używane w aplikacji, np. jako przykłady czy logo.

- **templates**  
  Folder przechowujący szablony HTML aplikacji webowej FastAPI. Wśród nich znajdują się: 
  - `index.html` – szablon strony głównej aplikacji.
  - `analyze_image.html` – szablon strony do przesyłania zdjęcia przekazywanego do analizy emocji.
  - `result.html` – szablon strony z wynikami przewidzianej emocji oraz zdjęcia, na którym to wykryto.
  - `camera.html` – szablon strony, na której możliwa jest analiza emocji z obrazu z kamery.
  - `about.html` – szablon strony z głównymi informacjami o aplikacji oraz autorze.
  - `error.html` – szablon strony, która informuje o błędzie w przesłanym zdjęciu lub przy analizie.

- **tests**  
  Folder z testami:
  - Folder `img`: Zawiera zdjęcia wykorzystywane do testowania aplikacji.
  - `test_main.py`: Skrypt zawierający testy jednostkowe, integracyjne, walidacyjne, systemowe oraz akceptacyjne dla `main.py`.

- **README.md**  
  Dokumentacja projektu, zawierająca informacje o instalacji, uruchamianiu oraz celach aplikacji.

- **haarcascade_frontalface_default.xml**  
  Plik używany do wykrywania twarzy przy pomocy algorytmu Haar Cascade.

- **main.py**  
  Główny kod aplikacji FastAPI:
  - Definiuje wszystkie endpointy.
  - Obsługuje przetwarzanie obrazów oraz logikę aplikacji.

- **model_CNN.h5**  
  Wstępnie nauczony model sieci konwolucyjnych (CNN) bazujący na zbiorze RAF-DB, stworzony przy użyciu kodu z pliku `complex_cnn.py`.

- **requirements.txt**  
  Plik definiujący wymagania dla projektu – lista bibliotek Python potrzebnych do działania aplikacji.

