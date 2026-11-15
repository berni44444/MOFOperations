# MOFOperations

`MOFOperations.py` ist das zentrale Modul fuer die Monte-Carlo-Optimierung funktionalisierter Metal-Organic Frameworks (MOFs). Es verbindet

- das Einlesen von PQ-XYZ-Strukturen,
- die Zuordnung von Atomen zu Residuen anhand eines Moldescriptors,
- zufaellige Rotations- und Permutationsbewegungen von Residuen,
- die Geometrieoptimierung mit ASE und MACE,
- die Metropolis-Akzeptanzentscheidung,
- Simulated Annealing ueber mehrere Thermocyclen und
- das Schreiben von Energie-, Trajektorien- und Strukturdateien.

Die hier beschriebene Klasse und die Hilfsfunktionen befinden sich in `MyOwnPackage/MOFOperations.py`.

## 1. Schnellstart: ein kompletter Thermocycle

Der normale Einstieg erfolgt ueber die Klasse `MOF`. Zuerst wird die YAML-Konfiguration geladen. Danach wird der Moldescriptor geladen, damit die Atome den passenden Residuen zugeordnet werden koennen. Anschliessend kann die Optimierung gestartet werden.

```python
from MOFOperations import MOF
from time import time

start = time()

mof = MOF()
mof.set_yaml("path/to/configuration.yaml")
mof.set_moldescriptor("path/to/moldescriptor.dat")

print("Setup complete. Starting Monte Carlo optimization...")
final_atoms = mof.thermocycle()

print(f"Optimierung beendet nach {time() - start:.2f} Sekunden.")
```

Dies fuehrt einen Thermocycle aus, bei dem die Haeufigkeit der Operationen (Permutation/Rotation) im yaml festgelegt wurden.
Die Parameter fuer die Thermocycles sind ebenfalls im yaml hinterlegt.

### Voraussetzungen

Das Modul verwendet unter anderem:

- Python,
- NumPy,
- PyYAML,
- ASE,
- `packaging` und
- `mace-torch` beziehungsweise `mace.calculators.mace_mp`.

Die MACE-Konfiguration im aktuellen Projekt verwendet `device="cuda"`. Fuer die gezeigte Konfiguration wird deshalb eine funktionierende CUDA-/PyTorch-Umgebung mit kompatiblem NVIDIA-Treiber benoetigt. Die Berechnung kann nicht automatisch auf CPU umschalten, weil das Geraet in `MOF.set_yaml()` fest auf CUDA gesetzt wird.

## Nachbarlisten

`CreateNeighborLists.py` kann Nachbarlisten als komprimierte NumPy-Datei (`.npz`) speichern. Dieses Format ist fuer die weitere Verarbeitung in Python besser geeignet als die optionale Textausgabe, da die Array-Struktur erhalten bleibt und keine eigene Textdatei geparst werden muss.

```python
from CreateNeighborLists import create_neighbor_list, load_neighbor_list

neighbors = load_neighbor_list("neighbors.npz")
orthogonal = neighbors[0]["orthogonal"]
in_line = neighbors[0]["in_line"]
```

Die drei Zeilen jedes Arrays enthalten den globalen Index des ersten Atoms, die Linker-Nummer und die Residuen-Nummer. Die `.npz`-Speicherung unterstuetzt auch unterschiedliche Nachbaranzahlen pro Linker. Wird in der YAML-Datei ein `output_file` mit der Endung `.npz` angegeben, wird automatisch dieses Format verwendet; andere Endungen erzeugen weiterhin die lesbare Textdatei.

Das Modul kann aus dem Paketordner oder nach Installation des Projekts importiert werden:

```bash
cd /path/to/MyOwnPackage
pip install -e .
```

Danach ist beispielsweise Folgendes moeglich:

```python
from MOFOperations import MOF
```

## 2. Wichtiger Ablauf der Initialisierung

Die Reihenfolge ist verbindlich:

```python
mof = MOF()
mof.set_yaml("configuration.yaml")
mof.set_moldescriptor("moldescriptor.dat")
```

### `MOF()`

Der Konstruktor erzeugt ein noch nicht konfiguriertes Optimierungsobjekt. Zu diesem Zeitpunkt sind weder eine Struktur noch ein Calculator geladen.

Intern werden ausserdem Vorlagen fuer eine Standardkonfiguration und einen Standard-Moldescriptor vorbereitet. Diese Vorlagen werden von `create_default_files()` verwendet.

### `set_yaml(filepath)`

`set_yaml()` erledigt mehrere Aufgaben:

1. Die YAML-Datei wird durch `readYaml()` eingelesen und validiert.
2. Dateinamen, Simulated-Annealing-Parameter und Bewegungsparameter werden in `mof` gespeichert.
3. Die PQ-XYZ-Struktur aus `filename` wird durch `pq_reader()` gelesen.
4. Ein MACE-Calculator wird erzeugt.
5. Der Calculator wird an `mof.atoms` angehaengt.

Der Eingabepfad `filename` muss auf eine PQ-XYZ-Datei zeigen. In diesem Projekt steht in der ersten Zeile nicht nur die Atomanzahl, sondern:

```text
N a b c alpha beta gamma
```

Die zweite Zeile wird von `pq_reader()` uebersprungen. Danach folgen die Atome mit Elementsymbol und drei kartesischen Koordinaten.

### `set_moldescriptor(filepath)`

Diese Methode muss nach `set_yaml()` aufgerufen werden. Sie:

1. liest den Moldescriptor mit `read_moldescriptor()` ein,
2. speichert die Molekueldefinitionen in `mof.moldescriptor` und
3. schreibt fuer jedes Atom `residueNumber` in `mof.atoms`.

Wird die Methode vor `set_yaml()` aufgerufen, wird ein `RuntimeError` ausgeloest.

## 3. YAML-Konfiguration

Eine Konfiguration beschreibt Eingabe- und Ausgabedateien, den Calculator, die Annealing-Parameter und die erlaubten Molekuelbewegungen.

Ein gekuerztes Beispiel:

```yaml
filename: "/pfad/zur/Initialstruktur.xyz"
outputfile: "/pfad/zur/trajectory.xyz"
logfile: "/pfad/zur/output.dat"
trajectoryfile: "/pfad/zur/trajectory.traj"
final_structurefile: "/pfad/zur/final_structure.xyz"

version: "medium"
Tfirst: 5000.0
Tinit: 200.0
Tfinal: 1.0
cooling_first: 0.995
cooling_rate: 0.99
number_of_thermocycles: 15

weight_permutation: 1.0
weight_rotation: 1.0
fmax: 0.1
max_steps: 10000
dispersion: true
hydrostatic_strain: true
bits: 32

rotational_residues:
  3: [10, 13, 14, 15]

angle_of_rotation: 180

permutational_residues:
  3: [10, 13, 14, 15]
  4: [8, 11, 12, 13]
```

### Eingabe- und Ausgabedateien

| Schluessel | Bedeutung |
|---|---|
| `filename` | PQ-XYZ-Eingabestruktur |
| `outputfile` | Plain-XYZ-Log der waehrend der Optimierung geschriebenen Strukturen |
| `logfile` | Energie-, Temperatur- und Akzeptanzstatistik |
| `trajectoryfile` | ASE-Binary-Trajektorie (`.traj`) |
| `final_structurefile` | Endstruktur nach den Thermocyclen |

Alle vier Log-/Ausgabepfade werden von `readYaml()` geprueft. Existiert eine der Dateien bereits, wird ein `FileExistsError` ausgeloest. Das verhindert, dass alte und neue Berechnungen vermischt werden. Vor einem neuen Lauf muessen deshalb neue Pfade gewaehlt oder alte Ergebnisdateien verschoben werden.

Die Ausgabepfade muessen ausserdem verschieden sein. Relative Pfade werden relativ zum aktuellen Arbeitsverzeichnis interpretiert, nicht automatisch relativ zum Speicherort der YAML-Datei.

### MACE und Optimierung

| Schluessel | Bedeutung |
|---|---|
| `version` | MACE-Modell, im Projekt z. B. `small`, `medium` oder `large` |
| `bits` | numerische Genauigkeit, nur `32` oder `64` |
| `dispersion` | Dispersion im MACE-Calculator aktivieren/deaktivieren |
| `hydrostatic_strain` | bei der Zelloptimierung nur hydrostatische Dehnung zulassen |
| `fmax` | Abbruchkriterium fuer `LBFGS` in eV/Angstrom |
| `max_steps` | maximale Anzahl Optimierungsschritte pro vorgeschlagener Bewegung |

### Simulated Annealing

Die Temperatur wird innerhalb eines Zyklus multiplikativ reduziert:

```text
T_neu = T_alt * cooling_rate
```

Der erste Zyklus beginnt mit `Tfirst` und verwendet `cooling_first`. Die anschliessenden Zyklen beginnen mit `Tinit` und verwenden `cooling_rate`. Jeder Zyklus endet, sobald `Tfinal` unterschritten wird.

`number_of_thermocycles` gibt die Anzahl der Zyklen nach dem ersten Hochtemperaturzyklus an. Bei `number_of_thermocycles: 15` werden daher insgesamt 16 Zyklen ausgefuehrt.

### Rotations- und Permutationsgewichte

```yaml
weight_rotation: 1.0
weight_permutation: 1.0
```

Die Gewichte bestimmen die relative Wahrscheinlichkeit der vorgeschlagenen Bewegungen in `__metropolis_step__()`:

```text
P(Permutation) = weight_permutation /
                 (weight_permutation + weight_rotation)
```

Eine Permutation benoetigt mindestens zwei verschiedene Eintraege in `permutational_residues`. Wird ihr Gewicht auf null gesetzt, wird sie deaktiviert. Analog wird die Rotation bei `weight_rotation: 0` deaktiviert.

## 4. Moldescriptor

Der Moldescriptor beschreibt die Atomreihenfolge jedes Molekuel- oder Residuentyps.

Der Kopf eines Molekuelblocks lautet:

```text
<Molekuelname> <Atomanzahl> <Gesamtladung>
```

Danach folgen genau so viele Atomzeilen wie angegeben:

```text
<Elementsymbol> <Atomtyp> <Partial Charge>
```

Ein Ausschnitt aus dem Projekt:

```text
Zn(II)        1       2
Zn            0       2.0

O(-II)        1      -2
O             0      -2.0

BDC-NOO2-    18      -2
C             0       0.0
N             0       0.0
O             0       0.0
...
```

Die `residueNumber` wird automatisch in der Reihenfolge der Molekueldefinitionen vergeben:

- erster Block: `residueNumber == 1`,
- zweiter Block: `residueNumber == 2`,
- dritter Block: `residueNumber == 3`,
- usw.

Die Schluessel in `rotational_residues` und `permutational_residues` beziehen sich auf diese Nummern, nicht auf die Namen des Moldescriptors.

Die Atomreihenfolge im Eingabe-XYZ muss mit einer der im Moldescriptor beschriebenen Sequenzen uebereinstimmen. `addResidueNumberToAtomsWithMoldescribtor()` sucht die laengsten Molekueldefinitionen zuerst und versucht dann, die komplette Atomliste von vorne nach hinten in Residuen aufzuteilen.

## 5. Die Klasse `MOF`

### `create_default_files(...)`

```python
configuration_path, moldescriptor_path = mof.create_default_files(
    directory="run_01",
    input_filename="initial.xyz",
    configuration_filename="configuration.yaml",
    moldescriptor_filename="moldescriptor.dat",
)
```

Erstellt eine Beispielkonfiguration und einen Beispiel-Moldescriptor. Die Eingabestruktur selbst wird nicht erzeugt. Standardmaessig werden vorhandene Dateien nicht ueberschrieben.

Wichtige Parameter:

- `directory`: Zielordner,
- `input_filename`: Name der Struktur, die in die YAML eingetragen wird,
- `configuration_filename`: Name der YAML-Datei,
- `moldescriptor_filename`: Name der Moldescriptor-Datei,
- `overwrite`: vorhandene Zieldateien bei `True` ersetzen.

Auf Modulebene existiert ausserdem ein gleichnamiger Wrapper:

```python
from MOFOperations import create_default_files

create_default_files("run_01")
```

### `LogXYZ()`

Schreibt den aktuellen Zustand von `mof.atoms` mit `LogToXYZ()` an `mof.outputfile`. Das Format ist das projektspezifische PQ-XYZ-Format. Die Methode wird im normalen Thermocycle intern verwendet.

### `rotate_random_residue()`

Fuehrt eine zufaellige Rotationsbewegung aus:

1. Ein konfigurierter Residuentyp wird zufaellig ausgewaehlt.
2. Ein konkretes Residuum dieses Typs wird zufaellig ausgewaehlt.
3. Das Residuum wird periodisch sinnvoll gewrappt.
4. Aus den konfigurierten lokalen Atomen werden Haupttraegheitsachsen bestimmt.
5. Eine Achse und anschliessend eine Rotation werden zufaellig ausgewaehlt.
6. Das Residuum wird in das Gesamtsystem zurueckgeschrieben und re-imaged.

Rueckgabewert ist das aktualisierte `ase.Atoms`-Objekt des Gesamtsystems, also `mof.atoms`.

### `swap_2_random_residues()`

Waehlt zwei verschiedene konfigurierte Permutations-Residuentypen aus, bestimmt daraus je ein konkretes Residuum und tauscht deren Positionen und Orientierungen. Die Orientierung wird aus einem lokalen Koordinatensystem bestimmt, das durch drei oder mehr konfigurierten Ankeratome definiert wird.

Rueckgabewert ist das aktualisierte Gesamtsystem.

### `thermocycle(...)`

Startet die komplette Monte-Carlo-/Simulated-Annealing-Rechnung:

1. aktuelle Struktur und Energie werden als Ausgangspunkt gespeichert,
2. eine Rotations- oder Permutationsbewegung wird vorgeschlagen,
3. die Struktur wird mit `optimize()` relaxiert,
4. die neue Energie wird berechnet,
5. die Bewegung wird nach Metropolis akzeptiert oder verworfen,
6. Temperatur und Statistik werden aktualisiert,
7. akzeptierte Strukturen werden in die Trajektorien geschrieben.

Optionale Parameter ueberschreiben die Pfade aus der YAML-Datei:

```python
final_atoms = mof.thermocycle(
    logfile="run_01/output.dat",
    trajectory_name="run_01/trajectory.traj",
    final_structure_name="run_01/final_structure.xyz",
)
```

Die Methode gibt die letzte Struktur als `ase.Atoms` zurueck. Am Ende wird die Struktur als Extended XYZ geschrieben und die ASE-Trajektorie geschlossen.

## 6. Zentrale freie Funktionen

Die freien Funktionen sind nuetzlich, wenn einzelne Schritte ausserhalb der `MOF`-Klasse untersucht oder getestet werden sollen.

### `readYaml(filepath)`

Liest und validiert eine YAML-Konfiguration. Zusaetzlich werden folgende Werte eingefuegt:

- `dtype`, entweder `float32` oder `float64`,
- `PermutationBool`,
- `RotationBool`.

Die Funktion prueft unter anderem Datentypen, positive Temperaturen, Cooling-Faktoren, Gewichte, vorhandene Ausgabedateien und die Anzahl der Permutations-Residuentypen.

```python
from MOFOperations import readYaml

parameters = readYaml("configuration.yaml")
print(parameters["dtype"])
print(parameters["RotationBool"])
```

### `pq_reader(filename)`

Liest eine PQ-XYZ-Datei und gibt ein ASE-`Atoms`-Objekt zurueck. Zelllaengen und Zellwinkel aus der ersten Zeile werden in `atoms.cell` gespeichert; periodische Randbedingungen werden fuer alle drei Richtungen aktiviert.

```python
from MOFOperations import pq_reader

atoms = pq_reader("Initial_optimized_MOF-5-NO2-0.5_reordered.xyz")
print(len(atoms))
print(atoms.cell.lengths())
```

### `read_moldescriptor(filenameMoldescriptor)`

Liest die Molekueldefinitionen und gibt eine Liste von Dictionaries zurueck:

```python
[
    {
        "name": "BDC-NOO2-",
        "atom_count": 18,
        "charge": -2.0,
        "elements": ["C", "N", "O", ...],
        "residueNumber": 3,
    },
]
```

### `addResidueNumberToAtomsWithMoldescribtor(moldescriptor, atoms)`

Ordnet jedem Atom anhand der Elementsymbol-Sequenz eine Residuenummer zu. Das Array kann danach wie folgt gelesen werden:

```python
residue_numbers = atoms.get_array("residueNumber")
```

Die Funktion veraendert das uebergebene `Atoms`-Objekt und gibt es ebenfalls zurueck.

### `extract_random_residue(atoms, molecules, residue_numbers_of_molecules_to_rotate)`

Waehlt zufaellig einen konfigurierten Residuentyp und eine konkrete Instanz dieses Typs aus.

Rueckgabe:

```python
residue, residue_type, global_indices = extract_random_residue(
    atoms,
    molecules,
    [3],
)
```

- `residue`: kopiertes ASE-`Atoms`-Objekt des ausgewaehlten Residues,
- `residue_type`: Moldescriptor-Nummer,
- `global_indices`: globale Atomindizes im Gesamtsystem.

#### Beispiel: `extract_random_residue`

```python
from MOFOperations import (
    addResidueNumberToAtomsWithMoldescribtor,
    extract_random_residue,
    pq_reader,
    read_moldescriptor,
)

atoms = pq_reader("initial.xyz")
molecules = read_moldescriptor("moldescriptor.dat")
atoms = addResidueNumberToAtomsWithMoldescribtor(molecules, atoms)

residue, residue_type, global_indices = extract_random_residue(
    atoms=atoms,
    molecules=molecules,
    residue_numbers_of_molecules_to_rotate=[3],
)

print("Typ:", residue_type)
print("Globale Indizes:", global_indices)
print("Anzahl Atome:", len(residue))
```

Voraussetzung ist, dass `atoms` bereits ein `residueNumber`-Array besitzt.

### `extract_2_random_residues(atoms, molecules, permutation_residue_data)`

Waehlt zwei verschiedene Residuentypen aus den Schluesseln von `permutation_residue_data` und je eine konkrete Instanz aus.

```python
residues, residue_types, global_indices = extract_2_random_residues(
    atoms,
    molecules,
    {3: [10, 13, 14, 15], 4: [8, 11, 12, 13]},
)
```

Die drei Listen sind positionsgleich: Index `0` in jeder Liste gehoert zum ersten Residuum, Index `1` zum zweiten. Es muessen mindestens zwei verschiedene Residuentypen vorhanden sein.

### `rotate_residue(residue, residue_number, rotation_residue_data, alpha=180)`

Rotiert alle Atome eines Residues um den Schwerpunkt der in `rotation_residue_data` angegebenen Atome.

Die Rotationsachse wird zufaellig aus den Hauptachsen des Traegheitstensors ausgewaehlt. Fuer `alpha == 180` wird eine spezielle, numerisch einfache 180-Grad-Rotation verwendet. Fuer andere Winkel wird die Rodrigues-Rotationsmatrix aufgebaut.

```python
rotated = rotate_residue(
    residue,
    residue_number=3,
    rotation_residue_data={3: [10, 13, 14, 15]},
    alpha=90,
)
```

Die Funktion veraendert das uebergebene `residue`-Objekt und gibt es zurueck.

### `LogToXYZ(atoms, filename)`

Haengt einen Strukturzustand an eine Datei an. Es wird das projektspezifische PQ-XYZ-Format geschrieben. Die Datei wird im Append-Modus geoeffnet.

```python
LogToXYZ(atoms, "trajectory.xyz")
```

Die Funktion schreibt keine Standard-XYZ-Kommentarzeile, sondern die Zellparameter direkt in die erste Zeile. Fuer Programme, die Standard-XYZ erwarten, sollte stattdessen der passende ASE-Writer verwendet werden.

### `write_acceptance_ratio(...)`

Schreibt eine formatierte Statistikzeile in eine bereits geoeffnete Logdatei und ruft anschliessend `flush()` auf. Die Parameter sind:

- aktuelle Energie,
- Temperatur,
- globale Akzeptanzrate,
- Rotationsversuche,
- Permutationsversuche,
- Rotationsakzeptanzrate,
- Permutationsakzeptanzrate.

Diese Funktion wird intern aus `thermocycle()` verwendet.

### `swap_residues(residues, vectors, centers)`

Tauscht zwei bereits extrahierte und vorbereitete Residuen geometrisch aus. Die lokalen Frames werden ueber eine Rotationsmatrix aufeinander abgebildet:

```python
swapped = swap_residues(
    residues=[residue_1, residue_2],
    vectors=[vectors_1, vectors_2],
    centers=[coaa_1, coaa_2],
)
```

`vectors[i]` muss die Form `(3, 3)` besitzen und die drei lokalen Vektoren enthalten:

```text
[vect_in_plane_1, vect_in_plane_2, normal]
```

Die Funktion schreibt die Residuen nicht automatisch in das Gesamtsystem zurueck und fuehrt kein Re-Imaging durch. Dafuer sind anschliessend `reImage2Residues()` und die globalen Atomindizes notwendig.

### `reImage1Residue(atoms, residue, global_indices)`

Verschiebt ein einzelnes vorbereitetes Residuum in die periodische Darstellung der Gesamtstruktur zurueck und schreibt dessen Positionen an den angegebenen globalen Indizes in `atoms`.

### `reImage2Residues(atoms, residues, global_indices)`

Entsprechende Funktion fuer zwei Residuen. Sie erwartet zwei Residuen und zwei Arrays globaler Indizes:

```python
atoms = reImage2Residues(
    atoms,
    [swapped_residue_1, swapped_residue_2],
    [indices_1, indices_2],
)
```

### `preparation_routine(linker_atoms, permutation_residue_data, residue_type, wrapped_to_coaa=False)`

Bereitet ein Residuum fuer Rotation oder Permutation vor:

1. Die Atome werden relativ zum ersten Atom periodisch gewrappt.
2. Die konfigurierten Ankeratome werden bestimmt.
3. Ihr geometrischer Mittelpunkt `coaa` wird berechnet.
4. Eine SVD der zentrierten Ankerpositionen liefert zwei In-Plane-Vektoren und den Normalenvektor.
5. Die drei Vektoren werden normiert.

Rueckgabe:

```python
wrapped_residue, coaa, vectors = preparation_routine(
    linker_atoms=residue,
    permutation_residue_data={3: [10, 13, 14, 15]},
    residue_type=3,
)
```

`vectors` hat die Form `(3, 3)`:

```text
vectors[0]  erster Vektor in der Ankerebene
vectors[1]  zweiter Vektor in der Ankerebene
vectors[2]  Normalenvektor
```

Mit `wrapped_to_coaa=True` werden zusaetzlich die Koordinaten des zurueckgegebenen Residues um `coaa` verschoben.

#### Beispiel: Residuum untersuchen

```python
residue, residue_type, indices = extract_random_residue(
    mof.atoms,
    mof.moldescriptor,
    list(mof.rotational_residues),
)

wrapped, coaa, vectors = preparation_routine(
    residue,
    mof.permutational_residues,
    residue_type,
)

print("Anchor center:", coaa)
print("Normal vector:", vectors[-1])
```

### `optimize(atoms, fmax=0.1, steps=200, hydrostatic_strain=True)`

Optimiert die Geometrie mit ASEs `LBFGS` und einem `UnitCellFilter`. Der Calculator muss bereits an `atoms.calc` gesetzt sein. Rueckgabe ist `True`, wenn die Optimierung konvergiert, ansonsten `False`.

```python
converged = optimize(
    mof.atoms,
    fmax=0.1,
    steps=1000,
    hydrostatic_strain=True,
)
```

## 7. Interner Ablauf einer Monte-Carlo-Bewegung

`MOF.__metropolis_step__(temperature)` ist intern und wird normalerweise nicht direkt aufgerufen. Der Ablauf ist:

```text
alte Struktur und alte Energie sichern
                 |
Rotation oder Permutation zufaellig waehlen
                 |
Geometrie mit LBFGS optimieren
                 |
bei Nichtkonvergenz: alte Struktur wiederherstellen
                 |
neue Energie berechnen
                 |
Metropolis-Wahrscheinlichkeit berechnen
                 |
Bewegung akzeptieren oder alte Struktur wiederherstellen
```

Die Akzeptanzwahrscheinlichkeit fuer eine energetisch unguenstigere Bewegung ist:

$$
P = \min\left(1, \exp\left[-\frac{E_{neu}-E_{alt}}{k_B T}\right]\right).
$$

Energetisch guenstigere Bewegungen werden immer akzeptiert. Bei einer abgelehnten Bewegung wird die alte Struktur inklusive Calculator wiederhergestellt.

## 8. Ausgabedateien

### Plain-XYZ-Log

`outputfile` wird durch `LogToXYZ()` fortlaufend erweitert. Akzeptierte Strukturen werden geschrieben; zu Beginn jedes Zyklus wird ebenfalls die aktuelle Struktur protokolliert.

### Energie- und Statistiklog

`logfile` enthaelt pro Zyklus einen Kopf und Zeilen mit:

- Energie,
- Temperatur,
- kumulativer Akzeptanzrate,
- Anzahl Rotationsversuche,
- Anzahl Permutationsversuche,
- Rotationsakzeptanzrate,
- Permutationsakzeptanzrate.

### ASE-Trajektorie

`trajectoryfile` wird mit `ase.io.Trajectory` geschrieben und ist fuer die weitere Analyse mit ASE geeignet.

### Finale Struktur

`final_structurefile` wird am Ende von `thermocycle()` als Extended XYZ geschrieben. Die Datei ist nicht dasselbe Format wie das durch `LogToXYZ()` geschriebene PQ-XYZ-Log.

## 9. Typische Fehler und ihre Ursachen

### `FileExistsError` beim Start

Eine konfigurierte Ausgabedatei existiert bereits. Neue Dateinamen verwenden oder alte Ergebnisdateien verschieben.

### `RuntimeError: Call set_yaml before set_moldescriptor`

`set_moldescriptor()` wurde vor `set_yaml()` aufgerufen.

### `KeyError` oder unvollstaendige Residuen

Die YAML-Schluessel, Residuentypen oder lokalen Atomindizes stimmen nicht mit dem Moldescriptor ueberein.

### Falsche `residueNumber`

Die Elementsymbol-Reihenfolge im Eingabe-XYZ passt nicht exakt zur Moldescriptor-Sequenz. Besonders wichtig sind auch die Atomanzahl und die Reihenfolge innerhalb jedes Residues.

### `ValueError` bei Permutation

Fuer eine Permutation muessen mindestens zwei verschiedene Residuentypen in `permutational_residues` vorhanden sein. Die angegebenen lokalen Indizes muessen innerhalb der jeweiligen Residuen liegen.

### MACE-/CUDA-Fehler

Pruefen, ob MACE installiert ist, PyTorch CUDA erkennt und der konfigurierte MACE-Modellname verfuegbar ist. `set_yaml()` fordert aktuell explizit `device="cuda"` an.

### Vorhandene oder vermischte Trajektorien

Alle Ausgabedateien muessen vor einem neuen Lauf eindeutig und nicht vorhanden sein. Besonders bei `LogToXYZ()` ist zu beachten, dass die Datei absichtlich im Append-Modus geoeffnet wird.

## 10. Minimaler manueller Workflow ohne `MOF.thermocycle()`

Die Einzelbausteine koennen auch separat untersucht werden:

```python
from MOFOperations import (
    addResidueNumberToAtomsWithMoldescribtor,
    extract_random_residue,
    optimize,
    pq_reader,
    preparation_routine,
    read_moldescriptor,
)

atoms = pq_reader("initial.xyz")
molecules = read_moldescriptor("moldescriptor.dat")
atoms = addResidueNumberToAtomsWithMoldescribtor(molecules, atoms)

residue, residue_type, global_indices = extract_random_residue(
    atoms,
    molecules,
    residue_numbers_of_molecules_to_rotate=[3],
)

wrapped, coaa, vectors = preparation_routine(
    residue,
    {3: [10, 13, 14, 15]},
    residue_type,
)

print("coaa:", coaa)
print("vectors:", vectors)
```

Fuer `optimize()` muss zuvor ein Calculator an `atoms.calc` gesetzt werden. Der komplette Calculator- und Monte-Carlo-Lebenszyklus wird deshalb normalerweise ueber die `MOF`-Klasse verwendet.

## 11. Verzeichnisbeispiel

Eine typische Projektstruktur sieht so aus:

```text
MasterArbeit/
|-- MyOwnPackage/
|   |-- MOFOperations.py
|   |-- README.md
|   |-- pyproject.toml
|-- MOF-5-NO2-0.5/
|   |-- MC_Optimization.py
|   |-- setup/
|       |-- configuration.yaml
|       |-- moldescriptorMOF-5-NO2.dat
|       |-- Initial_optimized_MOF-5-NO2-0.5_reordered.xyz
|       |-- MC_Data/
```

Der Start aus dem Beispielprojekt erfolgt beispielsweise so:

```bash
cd /MasterArbeit/MOF-5-NO2-0.5
python MC_Optimization.py
```

Dabei muss `MOFOperations` im Python-Suchpfad liegen, etwa durch eine editable Installation oder durch die bestehende Projektumgebung.
