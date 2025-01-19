# Wymagania i Instalacja

## Wymagania wstępne

- **Python**: 3.10.12
- **Conda** lub **venv** (dla środowisk wirtualnych)
    ```bash
    sudo apt-get install build-essential cmake
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libboost-all-dev
    ```
---

## Kroki instalacji

### 1. Utworzenie środowiska wirtualnego:

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

### 2. Instalacja wymaganych pakietów:
  ```bash
  pip install -r requirements.txt
  ```

## Uruchomienie serwisu do rozpoznawania emocji ze zdjęć i kamery:
  ```bash
  uvicorn main:app --reload
  ```

## Uruchomienie jednej z metod uczenia maszynowego w celach nauczenia modelu na zbiorze danych:
  Należy ściągnąć zbiór danych i podzielić na dwa katalogi: train oraz test. Następnie należy uruchomić jedną z metod w folderze src. 

  Najwyższe średnie wyniki uzyskanie na zbiorze RAF-DB to około 56% na danych testowych oraz 66% na danych treningowych.
  Najwyższe średnie wyniki uzyskanie na zbiorze CK+ to około 81% na danych testowych oraz 90% na danych treningowych.


# Opis struktury projektu
  Tu opisać wszystkie pliki i ich zawartości

# Opis testów aplikacji
Tu opisać wszystkie pliki i ich zawartości