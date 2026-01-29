Wykład 7 przenosi nas w świat zastosowań praktycznych – jak wymodelować "poszarpane" wykresy (np. notowania giełdowe, EKG, profile gór), mając tylko kilka punktów danych. Odpowiedzią są **Funkcje Interpolacji Fraktalnej (FIF)**.

Oto Twoja pigułka wiedzy z **Wykładu 7: Funkcje Interpolacji Fraktalnej**.

### I. Idea i Konstrukcja (FIF)

Mamy zestaw punktów danych , przez które musi przejść wykres. Zamiast łączyć je gładkimi wielomianami (jak w klasycznej numeryce), łączymy je strukturą samopodobną.

**Konstrukcja:**

1. Dzielimy odcinek czasu  na mniejsze przedziały.
2. Definiujemy przekształcenia afiniczne, które "kopiują" cały wykres w te mniejsze przedziały, odpowiednio go ściskając i przesuwając.
3. Kluczowe równanie (samopodobieństwo funkcji ):


* : ściska dziedzinę (czas).
* : **współczynnik skalowania pionowego**. To najważniejszy parametr! Decyduje o "szorstkości" wykresu.



### II. Istnienie Rozwiązania (Operator Read'a-Bajrakareviča)

Podobnie jak przy IFS, tutaj też działa magia punktu stałego.

* Zdefiniowany operator  (przekształcający funkcje) jest **kontrakcją** (zwężający), o ile współczynniki pionowe .
* **Wniosek:** Istnieje dokładnie jedna funkcja ciągła , która przechodzi przez zadane punkty i spełnia warunek fraktalny. Jej wykres jest atraktorem odpowiedniego systemu IFS .



### III. Wymiar Wykresu Funkcji (Wzory Egzaminacyjne!)

To jest klucz do sterowania "wyglądem" fraktala. Wymiar zależy od tego, jak mocno skalujemy wykres w pionie () w stosunku do poziomu.

Dla punktów równo odległych w czasie () i  przedziałów:

1. **Przypadek "Gładki":**
Jeśli suma pionowych skalowań jest mała (), to wykres jest mało poszarpany i ma wymiar ****.
2. **Przypadek Fraktalny:**
Jeśli suma jest duża () i punkty nie leżą na jednej prostej, to wymiar wynosi:


.



*Dla nierównych odstępów czasu stosuje się bardziej ogólny wzór uwikłany (Twierdzenie Barnsleya): .* 

### IV. Gładkość a Wymiar

Istnieje ścisły związek między "szorstkością" (wymiarem) a matematyczną gładkością (wykładnikiem Höldera ):


* Im wyższy wymiar  (bliższy 2), tym mniejsze  (bliskie 0)  wykres jest bardzo "nerwowy", ostry.
* Im niższy wymiar  (bliższy 1), tym większe  (bliskie 1)  wykres jest łagodniejszy.



### V. Nowe Funkcje (Nieliniowe)

Klasyczne FIF używają przekształceń afinicznych (linii prostych). Nowsze podejścia (omówione na końcu wykładu) wprowadzają elementy nieliniowe (np. sinusy) do funkcji , co pozwala modelować jeszcze bardziej złożone zjawiska, zachowując fraktalną strukturę .

---

### Szybki test wiedzy 🧠

Sprawdźmy rozumienie wpływu parametru .

**Masz dwa wykresy interpolacji fraktalnej oparte na tych samych punktach węzłowych. W pierwszym przypadku suma współczynników pionowych , a w drugim . Który wykres będzie wyglądał na bardziej "poszarpany" i dlaczego?**