#!/usr/bin/env python3

from pathlib import Path
from time import time
from matplotlib.table import Cell
import numpy as np

import yaml

import ase
from packaging import version 

from ase.atoms import Atoms
from ase.io import Trajectory, write
from ase.data import atomic_numbers
from ase.optimize import LBFGS
from ase.calculators.calculator import Calculator, all_changes
#from ase.calculators.mixing import SumCalculator
from ase.filters import UnitCellFilter
from ase.units import kB

import subprocess
import re
import shutil

import pandas as pd



if version.parse(ase.__version__) < version.parse("3.23.0"):
  from ase.constraints import ExpCellFilter
else:  
  from ase.filters import FrechetCellFilter 


start_time = time()


def readYaml(filepath: str):
    """
    Load and validate a Monte Carlo optimization configuration.

    Configured logging paths are checked before the calculation starts. Existing
    files and duplicate logging paths raise an exception because all logs are
    intended to describe one fresh run. Orientation, permutation, and dihedral
    logs are optional and disabled when their values are ``None``.

    Args:
        filepath (str): path to .yaml configuration file

    Returns:
        dict: Validated parameters plus ``dtype``, ``PermutationBool``, and
            ``RotationBool``.

    Raises:
        FileExistsError: If a configured logging file already exists.
        FileNotFoundError: If the input structure cannot be opened.
        TypeError, ValueError: If a configuration value is invalid.
    """


    # ========= load config file ============
    with open(filepath, "r") as f:
        paramsYaml = yaml.safe_load(f)


    if paramsYaml["folder"] == "":
        paramsYaml["folder"] = Path(filepath).parent
    
    if paramsYaml["outputfile"] == "":
        outputfile = "final_" + paramsYaml["filename"]
    else:
        outputfile = paramsYaml["outputfile"]

    paramsYaml["outputfile"] = outputfile

    prefix = paramsYaml.get("prefix", "")

    logging_files = {
        "outputfile": paramsYaml["folder"] + "/" + prefix + outputfile,
        "logfile": paramsYaml["folder"] + "/" + prefix + paramsYaml.get("logfile", "output.dat"),
        "trajectoryfile": paramsYaml["folder"] + "/" + prefix + paramsYaml.get("trajectoryfile", "traj_annealing.traj"),
        "final_structurefile": paramsYaml["folder"] + "/" + prefix + paramsYaml.get("final_structurefile", "final_structure.xyz"),
        "orientation_logfile": paramsYaml["folder"] + "/" + prefix + paramsYaml.get("orientation_logfile"),
        "permutation_logfile": paramsYaml["folder"] + "/" + prefix + paramsYaml.get("permutation_logfile"),
        "dihedral_logfile": paramsYaml["folder"] + "/" + prefix + paramsYaml.get("dihedral_logfile"),
        "optimization_logfile": paramsYaml["folder"] + "/" + prefix + paramsYaml.get("optimization_logfile"),
        "dftbp_temporary_optimization_log_file": paramsYaml["folder"] + "/" + prefix + paramsYaml.get("dftbp_temporary_optimization_log_file", "dftb.out")
    }
    
    for name, filename in logging_files.items():
        if filename is None:
            continue
        if not isinstance(filename, str) or not filename:
            raise TypeError(f"'{name}' must be a non-empty filename.")

        if Path(filename).exists():
            raise FileExistsError(
                f"Logging file already exists: '{filename}'. "
                "Please choose a different filename."
            )

    logging_paths = [
        Path(filename) for filename in logging_files.values() if filename is not None
    ]
    if len(logging_paths) != len(set(logging_paths)):
        raise ValueError("Logging files must have different filenames.")

    paramsYaml.update(logging_files)



    # bits
    if not isinstance(paramsYaml["bits"], int) or paramsYaml["bits"] not in (32, 64):
        raise ValueError(f"Invalid 'bits': {paramsYaml['bits']}. "
        "Allowed values are 32 or 64.")
    dtype = f"float{paramsYaml['bits']}"

    paramsYaml["dtype"] = dtype

    # dispersion
    if not isinstance(paramsYaml["dispersion"], bool):
        raise TypeError(f"Invalid 'dispersion': {paramsYaml['dispersion']}. "
        "Allowed values are true or false.")

    # hydrostatic_strain
    if not isinstance(paramsYaml["hydrostatic_strain"], bool):
        raise TypeError(f"Invalid 'hydrostatic_strain': {paramsYaml['hydrostatic_strain']}. "
                        "Allowed values are true or false.")

    # optimize cell parameters
    if not isinstance(paramsYaml["optimize_cell_parameters"], bool):
        raise TypeError(f"Invalid 'optimize_cell_parameters': {paramsYaml['optimize_cell_parameters']}. "
                        "Allowed values are true or false.")

    # mace version
    if not paramsYaml["version"].lower() in ("small", "medium", "large", "mace_dac"):
        raise ValueError(f"Invalid 'version': {paramsYaml['version']}. "
                         "Allowed values are 'small', 'medium', 'large' or 'mace_dac'.")

    if paramsYaml["version"].lower() == "mace_dac":
        paramsYaml["version"] = "/media/storage_5/bla/mace-models/mace-dac-1.model"

    
    

    # output frequency
    if (not isinstance(paramsYaml["output_frequency"], int)
            or isinstance(paramsYaml["output_frequency"], bool)):
        raise TypeError(f"Invalid 'output_frequency': {paramsYaml['output_frequency']}. "
                        "The value must be an integer.")

    paramsYaml["job_type"] = paramsYaml["job_type"].strip()

    if paramsYaml["job_type"] not in ("annealing", "sampling_constant_T", "annealing_dftbp", "sampling_constant_T_dftbp"):
        raise ValueError(f"Invalid 'job_type': {paramsYaml['job_type']}. "
                         "Allowed values are 'annealing' or 'sampling_constant_T' or 'annealing_dftbp' or 'sampling_constant_T_dftbp'.")


    if paramsYaml["job_type"] == "annealing" or paramsYaml["job_type"] == "annealing_dftbp":
        # Tfirst
        if not isinstance(paramsYaml["Tfirst"], float) or paramsYaml["Tfirst"] <= 0:
            raise TypeError(f"Invalid 'Tfirst': {paramsYaml['Tfirst']}. "
                            "The value must be a positive float.")

        # Tinit
        if not isinstance(paramsYaml["Tinit"], float) or paramsYaml["Tinit"] <= 0:
            raise TypeError(f"Invalid 'Tinit': {paramsYaml['Tinit']}. "
                            "The value must be a positive float.")


        # Tfinal
        if not isinstance(paramsYaml["Tfinal"], float) or paramsYaml["Tfinal"] <= 0:
            raise TypeError(f"Invalid 'Tfinal': {paramsYaml['Tfinal']}. "
                            "The value must be a positive float.")

        
        # first cooling rate
        if not isinstance(paramsYaml["cooling_first"], float) or not 0 < paramsYaml["cooling_first"] <= 1:
            raise TypeError(f"Invalid 'cooling_first': {paramsYaml['cooling_first']}. "
                            "The value must be greater than 0 and less or equal to 1.")

        # cooling rate
        if not isinstance(paramsYaml["cooling_rate"], float) or not 0 < paramsYaml["cooling_rate"] <= 1:
            raise TypeError(f"Invalid 'cooling_rate': {paramsYaml['cooling_rate']}. "
                            "The value must be greater than 0 and less or equal to 1.")


        # number of thermocycles
        if not isinstance(paramsYaml["number_of_thermocycles"], int) or paramsYaml["number_of_thermocycles"] <= 0:
            raise TypeError(f"Invalid 'thermocycles': {paramsYaml['number_of_thermocycles']}. "
                            "The value must be a positive integer.")

        # max_steps
        if not isinstance(paramsYaml["max_steps"], int) or paramsYaml["max_steps"]<=0:
            raise TypeError(f"Invalid 'max_steps': {paramsYaml['max_steps']}. "
                            "The value must be a positive integer.")


    else:
        # Sampling T
        if (not isinstance(paramsYaml["Tsampling"], float)
                or paramsYaml["Tsampling"] <= 0):
            raise TypeError(f"Invalid 'Tsampling': {paramsYaml['Tsampling']}. "
                            "The value must be a positive float or None.")

        # Sampling number of steps
        if not isinstance(paramsYaml["sampling_steps"], int) or paramsYaml["sampling_steps"] <= 0:
            raise TypeError(f"Invalid 'sampling_steps': {paramsYaml['sampling_steps']}. "
                            "The value must be a positive integer.")
        



    # weights
    if not isinstance(paramsYaml["weight_permutation"], float) or paramsYaml["weight_permutation"] < 0:
        raise TypeError(f"Invalid 'weight_permutation': {paramsYaml['weight_permutation']}. "
                        "The value must be a non-negative float.")

    if not isinstance(paramsYaml["weight_rotation"], float) or paramsYaml["weight_rotation"] < 0:
        raise TypeError(f"Invalid 'weight_rotation': {paramsYaml['weight_rotation']}. "
                        "The value must be a non-negative float.") 

    
    if not isinstance(paramsYaml["weight_dihedral_flips"], float) or paramsYaml["weight_dihedral_flips"] < 0:
        raise TypeError(f"Invalid 'weight_dihedral_flips': {paramsYaml['weight_dihedral_flips']}. "
                        "The value must be a non-negative float.")

    # Is some permutation wanted?
    permutation_residue_data = paramsYaml["permutational_residues"]

    if permutation_residue_data is None or paramsYaml["weight_permutation"] == 0:
        permutation = False
    elif len(permutation_residue_data) >= 2:
        permutation = True
    else:
        raise ValueError("Residue numbers for permutation must be >= 2 or empty!")

    # Atomic indices for plane orientation for permutation
    if permutation:
        for residue_type, atom_indices in permutation_residue_data.items():
            if not all(isinstance(index, int) for index in atom_indices):
                raise ValueError(f"Atomic indices for residue type {residue_type} "
                    "must be integers.")

    paramsYaml["PermutationBool"] = permutation


    # Is some rotation wanted?
    rotation_residue_data = paramsYaml["rotational_residues"]

    if rotation_residue_data is None or paramsYaml["weight_rotation"] == 0:
        rotation = False
    else:
        rotation = True

    paramsYaml["RotationBool"] = rotation

    # Atomic indices for axis for rotation
    if rotation:
        for residue_type, atom_indices in rotation_residue_data.items():
            if not all(isinstance(index, int) for index in atom_indices):
                raise ValueError(f"Atomic indices for residue type {residue_type} "
                    "must be integers.")

    # Is some dihedral flipping wanted?
    if paramsYaml["dihedral_residues"] is None or paramsYaml["weight_dihedral_flips"] == 0:
        paramsYaml["DihedralsBool"] = False
    else:
        paramsYaml["DihedralsBool"] = True

        for residue_type, dihedral_indices in paramsYaml["dihedral_residues"].items():
            if not all(isinstance(index, int)
                       for atom_indices in dihedral_indices
                       for index in atom_indices):
                raise ValueError(f"Atomic indices for residue type {residue_type} "
                                 "must be integers.")

    return paramsYaml




def _read_pq_frame(file) -> Atoms | None:
    """Read one PQ frame from an open text stream."""
    header = file.readline()
    if not header:
        return None

    fields = header.split()
    if len(fields) != 7:
        raise ValueError(f"Invalid PQ frame header: {header.rstrip()!r}")

    natoms, a, b, c, alpha, beta, gamma = map(float, fields)
    comment = file.readline()
    if not comment:
        raise EOFError("Unexpected end of PQ frame after header.")

    symbols = []
    positions = []
    for atom_index in range(int(natoms)):
        line = file.readline()
        if not line:
            raise EOFError(
                f"Unexpected end of PQ frame after {atom_index} of "
                f"{int(natoms)} atoms."
            )
        split_line = line.split()
        if len(split_line) < 4:
            raise ValueError(f"Invalid PQ atom line: {line.rstrip()!r}")
        symbols.append(split_line[0])
        positions.append([float(value) for value in split_line[1:4]])

    atoms = Atoms(symbols, positions=np.asarray(positions))
    atoms.set_cell([a, b, c, alpha, beta, gamma])
    atoms.set_pbc([True, True, True])
    return atoms


def pq_reader(filename: str) -> Atoms:
    """
    Read a .xyz file in the PQ format and return an Atoms object.

    Args:
        filename (str): path to the .xyz file

    Returns:
        Atoms: Atoms object
    """

    with open(filename, "r", encoding="utf-8") as file:
        return _read_pq_frame(file)



def read_moldescriptor(filenameMoldescriptor: str):
    """
    Read moldescriptor.dat

    Format:
        Name    Amount of atoms     Total Charge
        Element     (Atom type)    (MM point charge)
        Element     (Atom type)    (MM point charge)
        ...

    Returns:
        List of Dictionaries with:
            name
            atom_count
            charge (total)
            elements
            residueNumber
    """

    molecules = []

    with open(filenameMoldescriptor, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    residueNumber = 1

    while i < len(lines):

        line_number = i + 1
        # remove spaces in front (and end)
        line = lines[i].strip()
        i += 1

        # skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        parts = line.split()
   
        name = parts[0]

        try:
            atom_count = int(parts[1])
            charge = float(parts[2])
        except ValueError:
            print(
                f"Error in line {line_number}: "
                f"Amount of atoms or charge invalid."
            )
            continue

        elements = []

        # Jetzt die Elementzeilen lesen
        while i < len(lines):

            element_line = lines[i].strip()
            element_parts = element_line.split()

            if not element_parts:
                i += 1
                continue

            element = element_parts[0]

            # Ist es ein gültiges PSE-Element?
            if element in atomic_numbers:
                elements.append(element)
                i += 1

            else:
                # Kein Element -> neuer Molekülname.
                # Diese Zeile wird beim nächsten Schleifendurchlauf verarbeitet.
                break

        # Prüfen, ob Anzahl stimmt
        if len(elements) != atom_count:
            print(
                f"Error with '{name}' "
                f"(Line {line_number}): "
                f"mentioned are {atom_count} atoms, "
                f"only {len(elements)} were found."
            )

        molecules.append({
            "name": name,
            "atom_count": atom_count,
            "charge": charge,
            "elements": elements,
            "residueNumber": residueNumber
        })

        residueNumber += 1

    return molecules



def addResidueNumberToAtomsWithMoldescribtor(moldescriptor: list[dict], atoms: Atoms):
    """
    Add the residue numbers specified in the moldescriptor to the atoms object
    Args:
        - output of read_moldescriptor
        - atoms object
    Returns:
        - extended atoms object
    """

    # sort molecules by their length such that we don't accidentaly strip an O from it's middle
    molecules_sorted = sorted(moldescriptor, key=lambda x: len(x["elements"]), reverse=True)

    list_of_atoms_in_xyz = np.array(atoms.get_chemical_symbols())

    atoms.set_array("residueNumber", np.zeros(len(atoms), dtype=int))

    i = 0
    while i < len(list_of_atoms_in_xyz):

        found = False

        # Alle möglichen Molekültypen durchsuchen
        for molecule in molecules_sorted:

            elements = molecule["elements"]
            n = len(elements)
            
            # Passt die Molekül-Nummer ab Position i?
            if np.array_equal(list_of_atoms_in_xyz[i:i+n], elements):

                residueNumber = molecule["residueNumber"]

                # Alle Atome dieses Moleküls schreiben
                for j in range(n):

                    atom_index = i + j
                    atoms.arrays["residueNumber"][atom_index] = residueNumber

                i += n
                found = True
                break

        if not found:
            print(
                f"Fehler: Kein passendes Molekül für "
                f"Atom {i} ({list_of_atoms_in_xyz[i]}) gefunden."
            )
            i += 1


    return atoms



def LocalToGlobalIndices(atoms, local_indices, residue_number, moldescriptor):
    """
    Extracts the indices of atoms corresponding to nitro groups based on the provided moldescriptor.

    Parameters:
    - atoms: The ASE Atoms object containing the atomic structure.
    - local_indices: A list of local indices of atoms in the specified residue.
    - residue_number: The number of the residue to which the local indices belong.
    - moldescriptor: The moldescriptor data used to identify atom types.

    Returns:
    - A list of lists containing the indices of atoms corresponding to each nitro group.
    """
    residue_numbers = atoms.get_array("residueNumber")
    residue_length = moldescriptor[residue_number-1]["atom_count"]
    amount_of_residues = len(np.where(residue_numbers == residue_number)[0]) // residue_length
    
    global_indices = []

    # Calculate the starting index of the specified residue
    residue_start_index = np.where(residue_numbers == residue_number)[0][0]
    
    for i in range(amount_of_residues):
        # Calculate the starting index of the current residue
        current_residue_start_index = i * residue_length + residue_start_index
        # Convert local index to global index
        global_index = current_residue_start_index + np.array(local_indices)
        global_indices.append(global_index)

    return global_indices



def XYZ_to_RST(file_name_xyz: str, file_name_moldescriptor: str, output_file_name: str = ""):
    """
    Creates a rst file from an xyz file and a moldescriptor file
    Args:
        - file_name_xyz: str: 
            Path to the xyz file
        - file_name_moldecriptor: str: 
            Path to moldescriptor file
        - output_file_name (optional): str:
            Path to output file name (rst). 
            If not specified the filename will the xyz file name with rst end.
    """

    path_xyz = Path(file_name_xyz)
    folder_xyz = path_xyz.parent

    if output_file_name == "":
        output_file = folder_xyz / (path_xyz.stem + ".rst")
    else:
        output_file = output_file_name

    if Path(output_file).exists():
        raise FileExistsError(
            f"Output file already exists: '{output_file}'. "
            "Please choose a different outputfile."
        )


    atoms = pq_reader(file_name_xyz)

    list_of_atoms_in_xyz = np.array(atoms.get_chemical_symbols())
    box = np.append(atoms.cell.lengths(), atoms.cell.angles())
    coordinates = atoms.get_positions().tolist()


    
    molecules = read_moldescriptor(file_name_moldescriptor)
    # sort molecules by their length such that we don't accidentaly strip an O from it's middle
    molecules_sorted = sorted(molecules, key=lambda x: len(x["elements"]), reverse=True)
    

    output_lines = np.zeros((len(list_of_atoms_in_xyz),9), dtype=object)

    i = 0

    while i < len(list_of_atoms_in_xyz):

        found = False

        # Alle möglichen Molekültypen durchsuchen
        for molecule in molecules_sorted:

            elements = molecule["elements"]
            n = len(elements)

            # Passt der Molekül-Nummer ab Position i?
            if np.array_equal(list_of_atoms_in_xyz[i:i+n], elements):

                residueNumber = molecule["residueNumber"]

                # Alle Atome dieses Moleküls schreiben
                for j in range(n):

                    atom_index = i + j
                    element = list_of_atoms_in_xyz[atom_index]

                    output_lines[atom_index] = [element, 
                                                atom_index+1, residueNumber, 
                                                coordinates[atom_index][0], coordinates[atom_index][1], coordinates[atom_index][2], 
                                                0,0,0]

                i += n
                found = True
                break

        if not found:
            print(
                f"Fehler: Kein passendes Molekül für "
                f"Atom {i} ({list_of_atoms_in_xyz[i]}) gefunden."
            )
            i += 1



    

    print(output_lines[0])
    with open(output_file, "w") as f:

        # Schritt
        f.write("Step 0\n")

        # Box
        f.write(
            f"Box {box[0]:.10f} {box[1]:.10f} {box[2]:.10f} "
            f"{box[3]:.6f} {box[4]:.6f} {box[5]:.6f}\n"
        )

        # Atome
        # Format:
        # Atom Laufindex Residuennummer x y z 0 0 0

        
        for line in output_lines:

            f.write(f"{line[0]:5s} {line[1]} {line[2]} " # element + index + residue number 
                    f"{line[3]:.8f} {line[4]:.8f} {line[5]:.8f} " # xyz
                    f"{line[6]} {line[7]} {line[8]}\n") # 0

    print(f"\n\nDone. RST File created in {output_file}\n\n")

            
def extract_random_residue(atoms: Atoms, 
                           molecules: list, 
                           residue_numbers_of_molecules_to_rotate: list):
    """
    Selects a random residue of one of the specified residue types.
    
    Args:
        - atoms: ase:Atoms object of all atoms in the xyz file
        - molecules: dictionary output of read_moldescriptor
        - residue_numbers_of_molecules_to_rotate: list of residue numbers (int) that can be selected for rotation

    Returns:
        -rotating_residue : ase.Atoms
            Randomly selected residue.
        -residue_number : int
            Residue type number.
        -global_indices : numpy array
            Array of global indices of the residue atoms
    """

    residue_number_to_rotate = np.random.choice(residue_numbers_of_molecules_to_rotate)

    length = next(molecule["atom_count"] for molecule in molecules 
                  if molecule["residueNumber"] == residue_number_to_rotate)

    residue_numbers = atoms.get_array("residueNumber")

    indices = np.where(residue_numbers == residue_number_to_rotate)[0]

    # Auf einzelne zusammenhängende Residuen aufteilen
    residues = np.split(indices, np.arange(length, len(indices), length))

    # Kontrolle
    if not all(len(residue) == length for residue in residues):
        raise ValueError(
            f"Mindestens ein Residue des Typs {residue_number_to_rotate} "
            f"hat nicht die erwartete Länge {length}."
        )

    # Zufällig eine tatsächliche Residue auswählen
    selected_indices = residues[
        np.random.randint(len(residues))
    ]

    return atoms[selected_indices], residue_number_to_rotate, selected_indices



def extract_2_random_residues(atoms: Atoms,
                              molecules: list[dict],
                              permutation_residue_data: dict):
    """
    Select two random residues of two different allowed residue types.

    Args:
        atoms : ase.Atoms
            ASE Atoms object containing the complete system.

        molecules : list[dict]
            List of molecule dictionaries as returned by
            ``read_moldescriptor``. Each dictionary must contain
            ``"residueNumber"`` and ``"atom_count"``.

        permutation_residue_data : dict
            Dictionary defining the residue types that are allowed
            for permutation. The keys are the residue type numbers.
            The corresponding values contain the local atom indices
            used later to define the orientation plane of each residue.

            Example::

                {
                    3: [9, 12, 13],
                    8: [7, 10, 11]
                }

    Returns:
        tuple
            A tuple containing three lists:

            ``residues`` : list of ase.Atoms
                The two randomly selected residues.

            ``residue_types`` : list of int
                The residue type number corresponding to each selected
                residue.

            ``global_indices`` : list of numpy.ndarray
                The global atom indices of each selected residue.

            The entries in all three lists correspond to each other.
            For example::

                residues[0]
                residue_types[0]
                global_indices[0]

            all describe the first selected residue.

    Raises:
        ValueError
            If fewer than two different residue types are available
            for permutation.

            Also raised if one or more residues do not have the
            expected number of atoms according to ``molecules``.
    """

    # Get the residue types that are allowed for permutation
    allowed_residue_types = np.array( list(permutation_residue_data.keys()) )

    if len(allowed_residue_types) < 2:
        raise ValueError("At least two different residue types must be "
                         "available for permutation.")

    # Randomly select two different residue types
    chosen_residue_types = np.random.choice(allowed_residue_types, size=2, replace=False)

    residue_type_1, residue_type_2 = chosen_residue_types

    # Determine the expected number of atoms for each residue type
    length_1 = next(molecule["atom_count"] for molecule in molecules
        if molecule["residueNumber"] == residue_type_1)

    length_2 = next(molecule["atom_count"] for molecule in molecules
        if molecule["residueNumber"] == residue_type_2)

    # Get the residue type of every atom in the complete system
    residue_numbers = atoms.get_array("residueNumber")

    indices_1 = np.where(residue_numbers == residue_type_1)[0]

    indices_2 = np.where(residue_numbers == residue_type_2)[0]

    # Split the global indices into individual contiguous residues
    residues_1 = np.split(indices_1, np.arange(length_1, len(indices_1), length_1))

    residues_2 = np.split(indices_2, np.arange(length_2, len(indices_2), length_2))


    # Check that every residue has the expected number of atoms
    if not all(len(residue) == length_1 for residue in residues_1):
        raise ValueError(f"At least one residue of type {residue_type_1} "
                         f"does not have the expected length {length_1}.")

    if not all(len(residue) == length_2 for residue in residues_2):
        raise ValueError(f"At least one residue of type {residue_type_2} "
                         f"does not have the expected length {length_2}.")

    # Randomly select one residue instance of each type
    selected_indices_1 = residues_1[ np.random.randint(len(residues_1)) ]

    selected_indices_2 = residues_2[ np.random.randint(len(residues_2)) ]

    # Extract the corresponding ASE Atoms objects
    residue_1 = atoms[selected_indices_1]
    residue_2 = atoms[selected_indices_2]

    # Group the results into lists.
    # The same index in each list refers to the same residue.
    residues = [residue_1, residue_2]

    residue_types = [residue_type_1, residue_type_2]

    global_indices = [selected_indices_1, selected_indices_2]

    return residues, residue_types, global_indices






 
def rotate_residue(residue: Atoms, 
                   residue_number: int, 
                   dihedral_residue_data: dict, 
                   alpha: float=180):
    """
    Args:

        residue : ase.Atoms
            The residue to be rotated.

        residue_number : int
            Residue number of the selected residue.

        dihedral_residue_data : dict
            dihedral_residue_data is a dictionary that maps each residue type to the local atom indices defining its dihedral angles.
                Example: {3: [1,2,3,4], 4: [8,11,12,13]}

                The keys are residue numbers, and the values are lists of local atom indices within the corresponding residue
        
        alpha, optional : float
            angle (in deg) for rotation, default = 180

    Returns:
        residue : ase.Atoms
            Updated atoms object with new coordinates
        axis_index : int
            Index of the chosen rotational axis
            This is the index of the principal axis that was chosen for the rotation.
            0 corresponds to the axis with the smallest moment of inertia (axial), 
            1 to the intermediate axis (plane), 
            and 2 to the axis with the largest moment of inertia (normal).
    """


    # Select the atom indices
    selected_indices = np.asarray(dihedral_residue_data[residue_number], dtype=int)

    # Extract the corresponding atoms
    selected_atoms = residue[selected_indices]

    # Calculate center of mass
    com = selected_atoms.get_center_of_mass()

    # Positions relative to COM
    positions = selected_atoms.get_positions() - com
    masses = selected_atoms.get_masses()

    # =========== calculate the main rotational axes of inertia =============

    # Calculate inertia tensor
    inertia_tensor = np.zeros((3, 3))

    for mass, r in zip(masses, positions):
        x, y, z = r

        inertia_tensor += mass * np.array([
            [y**2 + z**2, -x*y,       -x*z],
            [-x*y,        x**2 + z**2, -y*z],
            [-x*z,        -y*z,        x**2 + y**2]
        ])

    # Principal axes
    moments, principal_axes = np.linalg.eigh(inertia_tensor)
    # sort the principal axes by the moments of inertia
    sorted_indices = np.argsort(moments)
    principal_axes = principal_axes[:, sorted_indices]



    # =========== randomly choose one of the axes =============
    axis_index = np.random.randint(3)
    rotation_axis = principal_axes[:, axis_index]

    # ========== rotational matrix ==========

    if alpha==180:
        rotation_matrix = 2*np.outer(rotation_axis, rotation_axis) - np.eye(3)

    elif alpha%360 ==0:
        rotation_matrix = np.eye(3)

    else:
        alpha_rad = np.deg2rad(alpha)
        cos_alpha = np.cos(alpha_rad)
        sin_alpha = np.sin(alpha_rad)
        n1 = rotation_axis[0]
        n2 = rotation_axis[1]
        n3 = rotation_axis[2]

        rotation_matrix = [[n1**2*(1-cos_alpha)+cos_alpha,      n1*n2*(1-cos_alpha)-n3*sin_alpha,   n1*n3*(1-cos_alpha)+n2*sin_alpha],
                        [n1*n2*(1-cos_alpha)+n3*sin_alpha,   n2**2*(1-cos_alpha)+cos_alpha,      n2*n3*(1-cos_alpha)-n1*sin_alpha],
                        [n1*n3*(1-cos_alpha)-n2*sin_alpha,   n2*n3*(1-cos_alpha)+n1*sin_alpha,   n3**2*(1-cos_alpha)+cos_alpha]]

        rotation_matrix = np.array(rotation_matrix)

    # Rotate ALL atoms of the residue around the COM
    residue_positions = residue.get_positions()
    relative_positions = residue_positions - com

    #rotated_positions = (rotation_matrix @ relative_positions + com)
    rotated_positions = (relative_positions@rotation_matrix.T + com)

    # Copy complete system
    #atoms_new = atoms.copy()

    # Replace only the coordinates of the selected residue
    #atoms_new.positions[np.asarray(residue_indices)] = rotated_positions
    residue.set_positions(rotated_positions)

    return residue, axis_index


def Dihedral_Flipping(linker: Atoms,
                      residue_number: int,
                      dihedral_residue_data: dict) -> Atoms:
    """Mirror all non-anchor atoms of a linker through its anchor plane.

    ``dihedral_residue_data`` maps each residue type to two index lists:
    ``[flipping_indices, anchor_indices]``. The flipping indices define the
    central-ring plane, while the anchor indices define the plane through which
    all non-anchor atoms are reflected.

    Args:
        linker : ase.Atoms
            The linker to be flipped.
        residue_number : int
            Residue number of the selected linker.
        dihedral_residue_data : dict
            Dictionary mapping residue types to flipping and anchor atom indices.
            Example::
                {
                    3: [[0, 1, 2], [3, 4, 5]],
                    4: [[6, 7, 8], [9, 10, 11]]
                }
    
    Returns:
        flipped_linker : ase.Atoms
            The linker with all non-anchor atoms reflected through the anchor plane.
    """

    if residue_number not in dihedral_residue_data:
        raise ValueError(f"No dihedral atom indices are defined for residue "
                         f"type {residue_number}.")

    residue_indices = dihedral_residue_data[residue_number]
    if len(residue_indices) != 2:
        raise ValueError("Dihedral residue data must contain flipping and "
                         "anchor atom indices.")

    flipping_indices = np.asarray(residue_indices[0], dtype=int)
    anchor_indices = np.asarray(residue_indices[1], dtype=int)

    for name, indices in (("flipping", flipping_indices),
                          ("anchor", anchor_indices)):
        if indices.ndim != 1 or len(indices) < 3:
            raise ValueError(f"At least three {name} atom indices are required.")
        if np.any(indices < 0) or np.any(indices >= len(linker)):
            raise ValueError(f"At least one {name} atom index is outside the linker.")

    anchor_plane_center, anchor_plane_vectors = SVD_plane_fit(linker[anchor_indices])
    anchor_plane_normal = anchor_plane_vectors[2]

    anchor_set = set(anchor_indices.tolist())
    chosen_indices = np.array(
        [index for index in range(len(linker)) if index not in anchor_set],
        dtype=int,
    )

    flipped_linker = linker.copy()
    relative_positions = flipped_linker[chosen_indices].get_positions() - anchor_plane_center
    reflected_positions = (
        relative_positions
        - 2.0 * np.outer(relative_positions @ anchor_plane_normal,
                         anchor_plane_normal)
        + anchor_plane_center
    )
    flipped_linker.positions[chosen_indices] = reflected_positions

    return flipped_linker



def LogToXYZ(atoms, filename, mode="a"):
    """Append an ASE Atoms object to an XYZ file."""

    # Number of atoms
    n_atoms = len(atoms)

    # Lattice parameters
    a, b, c = atoms.cell.lengths()
    alpha, beta, gamma = atoms.cell.angles()

    # Open in append mode
    with open(filename, mode) as f:

        # First line
        f.write(
            f"{n_atoms} "
            f"{a:.6f} {b:.6f} {c:.6f} "
            f"{alpha:.6f} {beta:.6f} {gamma:.6f}\n"
        )

        # Empty line
        f.write("\n")

        # Atom data
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        for symbol, (x, y, z) in zip(symbols, positions):
            f.write(
                f"{symbol:5s} "
                f"{x:.8f} {y:.8f} {z:.8f}\n"
            )




def LogOptimizationSteps(filename: str | None, MC_step: int, optimization_steps: int, operation: str) -> None:
    if filename is not None:
        with open(filename, "a") as f:
            if f.tell() == 0:
                f.write("MC_steps\tOptimizer_steps\tOperation\n")
            f.write(f"{MC_step} \t {optimization_steps} \t{operation}\n")

    


def update_rotation(orientation: np.ndarray, rotationalVector: int, linker_number: int)-> np.ndarray:
    """
    Update the orientation log for a linker based on the rotation vector.
    Args:
        orientation : np.ndarray
            Current orientation log for all linkers (binary representation).
        rotationalVector : int
            The rotation vector indicating the type of rotation applied.
            0: Rotate around the axial axis (smallest moment of inertia).
            1: Rotate around the plane axis (intermediate moment of inertia).
            2: Rotate around the normal axis (largest moment of inertia).
        linker_number : int
            The index of the linker whose orientation is being updated.
    Returns:
        np.ndarray
            Updated orientation log for all linkers after applying the rotation.
    """
    match rotationalVector:
        case 0: orientation[linker_number] ^= 0b01
        case 1: orientation[linker_number] ^= 0b10
        case 2: orientation[linker_number] ^= 0b11

    return orientation




def update_dihedral(dihedral: np.ndarray, linker_number: int) -> np.ndarray:
    """
    Update the dihedral log for a linker.

    Args:
        dihedral : np.ndarray
            Current dihedral log for all linkers (binary representation).
        linker_number : int
            The index of the linker whose dihedral is being updated.

    Returns:
        np.ndarray
            Updated dihedral log for all linkers after applying the update.
    """

    dihedral[linker_number] ^= 0b1

    return dihedral



def update_permutation(permutation: np.ndarray, residue_number_1: int, residue_number_2: int) -> np.ndarray:
    """
    Update the permutation log for two residues.

    Args:
        permutation : np.ndarray
            Current permutation log for all residues.
        residue_number_1 : int
            The index of the first residue whose permutation is being updated.
        residue_number_2 : int
            The index of the second residue whose permutation is being updated.

    Returns:
        np.ndarray
            Updated permutation log for all residues after applying the update.
    """

    # Update the permutation log for the two residues
    temp = permutation[residue_number_1]
    permutation[residue_number_1] = permutation[residue_number_2]
    permutation[residue_number_2] = temp

    return permutation




def write_acceptance_ratio(f,
                           steps: int,
                           time: float,
                           energy: float,
                           temperature: float,
                           ratio: float,
                           rotation_attempts: int,
                           permutation_attempts: int,
                           dihedral_attempts: int,
                           rotation_ratio: float,
                           permutation_ratio: float,
                           dihedral_ratio: float):
    """Write one simulated-annealing record with operation statistics."""

    f.write(
        f"\t\t{steps}\t{time:.3f}"
        f"\t{energy:.5f}\t{temperature:.3f}\t\t{ratio:.4f}"
        f"\t{rotation_attempts}\t{permutation_attempts}"
        f"\t{dihedral_attempts}"
        f"\t{rotation_ratio:.4f}\t{permutation_ratio:.4f}"
        f"\t{dihedral_ratio:.4f}\n"
    )
    f.flush()






def append_array_log(filepath: str | None, values: np.ndarray, step: int=0) -> None:
    """Append one integer-array snapshot without touching disabled logs."""
    if filepath is None:
        return
    with open(filepath, "a") as log_file:
        np.savetxt(log_file, np.asarray( np.append([step],values), dtype=int)[None, :], fmt="%d")


def swap_residues(residues: list[Atoms],
                  vectors: list[np.ndarray],
                  centers: list[np.ndarray]) -> list[Atoms]:
    """
    Exchange the positions and orientations of two residues.

    The local coordinate frame of residue 1 is mapped onto the
    local coordinate frame of residue 2, and vice versa.

    The center of the anchor atoms is used as the origin of the
    local coordinate frame and as the translation point.

    Args:

        residues : list[ase.Atoms]
            The two residues.

        vectors : list[np.ndarray]
            The two local coordinate frames.

            Each element must have shape (3, 3):

                [vec1,
                 vec2,
                 normal]

            where the vectors are orthonormal and correspond to
            the residue's local orientation.

        centers : list[np.ndarray]
            Center of the anchor atoms for each residue.


    Returns:
        swapped_residues : tuple (ase.Atoms, ase.Atoms)
            Swapped residues, not yet re-imaged
    """

    # ---------------------------------------------------------------
    # Check input
    # ---------------------------------------------------------------

    if len(residues) != 2:
        raise ValueError("Exactly two residues must be provided.")

    if len(vectors) != 2:
        raise ValueError("Exactly two vector frames must be provided.")

    if len(centers) != 2:
        raise ValueError("Exactly two centers must be provided.")


    # Copy residues
    swapped_residue_1 = residues[0].copy()
    swapped_residue_2 = residues[1].copy()


    # ---------------------------------------------------------------
    # Extract frames and centers
    # ---------------------------------------------------------------
    
    frame_1 = np.asarray(vectors[0], dtype=float).T
    frame_2 = np.asarray(vectors[1], dtype=float).T

    center_1 = np.asarray(centers[0], dtype=float)
    center_2 = np.asarray(centers[1], dtype=float)

    if frame_1.shape != (3, 3) or frame_2.shape != (3, 3):
        raise ValueError("Each vector frame must have shape (3, 3).")

    # ---------------------------------------------------------------
    # Calculate rotations between the complete frames
    # ---------------------------------------------------------------

    # Frame 1 -> Frame 2
    rotation_1_to_2 = frame_2 @ frame_1.T
    
    # Frame 2 -> Frame 1
    rotation_2_to_1 = frame_1 @ frame_2.T


    # ---------------------------------------------------------------
    # Transform residue 1
    # ---------------------------------------------------------------

    positions_1 = swapped_residue_1.get_positions()

    centered_positions_1 = positions_1 - center_1

    new_positions_1 = (centered_positions_1 @ rotation_1_to_2.T + center_2)

    # ---------------------------------------------------------------
    # Transform residue 2
    # ---------------------------------------------------------------

    positions_2 = swapped_residue_2.get_positions()

    centered_positions_2 = positions_2 - center_2

    new_positions_2 = (centered_positions_2 @ rotation_2_to_1.T + center_1)

    # ---------------------------------------------------------------
    # Write coordinates back
    # ---------------------------------------------------------------

    swapped_residue_1.set_positions(new_positions_1)
    swapped_residue_2.set_positions(new_positions_2)


    return [swapped_residue_1, swapped_residue_2]



def reImage1Residue(atoms : Atoms,
                    residue : Atoms,
                    global_indices : list[np.ndarray]):


    Lx, Ly, Lz = atoms.cell.lengths()
    cell_lengths = np.array([Lx, Ly, Lz])

    positions = residue.get_positions()

    reimaged_positions = positions - cell_lengths*np.round(positions/cell_lengths)
    
    atoms.positions[global_indices] = reimaged_positions

    return atoms



def reImage2Residues(atoms : Atoms, 
            residues : list[Atoms],
            global_indices : list[np.ndarray]):

    residue1, residue2 = residues

    Lx, Ly, Lz = atoms.cell.lengths()
    cell_lengths = np.array([Lx, Ly, Lz])

    positions1 = residue1.get_positions()
    positions2 = residue2.get_positions()    

    reimaged_positions1 = positions1 - cell_lengths*np.round(positions1/cell_lengths)
    reimaged_positions2 = positions2 - cell_lengths*np.round(positions2/cell_lengths)

    atoms.positions[global_indices[0]] = reimaged_positions1
    atoms.positions[global_indices[1]] = reimaged_positions2

    return atoms




def SVD_plane_fit(atoms: Atoms):
    """
    Fit a plane through the atoms using Singular Value Decomposition (SVD).

    Args:
        atoms : ase.Atoms
            ASE Atoms object containing the atoms to fit the plane through.

    Returns:
        tuple
            ``center`` : numpy.ndarray
                Geometrical center of the atoms as a Cartesian coordinate with shape ``(3,)``.

            ``vectors`` : numpy.ndarray
                Vectors of the fitted plane in the shape 
                [vec_in_plane_1, vec_in_plane_2, normal_vec]
    """

    positions = atoms.get_positions()

    # Calculate geometrical center
    center = np.mean(positions, axis=0)

    # Center the positions
    centered_positions = positions - center

    # Perform SVD
    U, S, Vt = np.linalg.svd(centered_positions, full_matrices=False)

    # Extract vectors
    vec1 = Vt[0, :]  # first in-plane vector
    vec2 = Vt[1, :]  # second in-plane vector
    vecNormal = Vt[2, :]  # normal vector of plane

    # Normalize them
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    vecNormal /= np.linalg.norm(vecNormal)

    vectors = np.array([vec1, vec2, vecNormal])

    return center, vectors



def normal_vector_check_and_corrections(vectors: np.ndarray, residue:Atoms, indicesForReferenceVector: list[int]):
    """
    Check if the normal vector of the fitted plane is pointing in the same direction as the 
    reference vector defined by the three anchor atoms. If not, flip the normal vector.
    
    Args:
        vectors : numpy.ndarray
            Array of vectors representing the fitted plane.
        residue : ase.Atoms
            ASE Atoms object containing the atoms.
        indicesForReferenceVector : list[int]
            List of indices for the three anchor atoms used to define the reference vector.

    Returns:
        numpy.ndarray
            The corrected vectors.
    """

    normal_vector = vectors[2]
    positions = residue[indicesForReferenceVector].get_positions()
    reference_vector = np.cross( (positions[1] - positions[0]), (positions[2] - positions[0]) )
    reference_vector /= np.linalg.norm(reference_vector)

    if np.dot(normal_vector, reference_vector) < 0:
        # Flip the normal vector
        vectors[2] = -normal_vector

    return vectors




def preparation_routine(linker_atoms: Atoms,
                        permutation_residue_data: dict,
                        residue_type: int,
                        wrapped_to_coaa: bool = False):
    """
    Wrap a linker relative to the
    geometrical center of its anchor atoms (coaa).

    The first atom of ``linker_atoms`` (local index 0) is used as
    the reference position. For every atom, the coordinates are
    shifted by an integer multiple of the corresponding lattice
    parameter such that the atom is placed in the periodic image
    closest to the first atom. 
    Then all coordinates are shifted to coaa if desired.

    For each Cartesian direction:

        x_new = x_i - Lx * round((x_i - x_0) / Lx)

    and analogously for y and z.

    The anchor atoms are then determined from
    ``permutation_residue_data`` using ``residue_type``. Their
    geometrical center is calculated from the wrapped coordinates.

    Lastly a plane is fitted through the anchor atoms with SVD
    and its vectors are returned.

    Args:
        linker_atoms : ase.Atoms
            ASE Atoms object containing the linker.

        permutation_residue_data : dict
            Dictionary mapping residue types to the local atom
            indices defining the anchor plane.

            Example::

                {
                    3: [9, 12, 13],
                    8: [7, 10, 11]
                }

        residue_type : int
            Residue type of the linker. This determines which
            atoms are used as anchor atoms.
        
        wrapped_to_coaa : bool, optional
            Whether the returned atoms object should be centered
            around the coaa or not. Default is False.

    Returns:
        tuple
            ``wrapped_linker`` : ase.Atoms
                Copy of ``linker_atoms`` with the coordinates wrapped
                relative to center of anchor atoms.

            ``center_of_anchor_atoms`` : numpy.ndarray
                Geometrical center of the three anchor atoms as a
                Cartesian coordinate with shape ``(3,)``.

            ``vectors`` : numpy.ndarray
                Vectors of the fitted plane in the shape 
                [vec_in_plane_1, vec_in_plane_2, normal_vec]

    Raises:
        ValueError
            If ``linker_atoms`` is empty, if the residue type is not
            present in ``permutation_residue_data``, or if the
            specified anchor indices are invalid.
    """

    if len(linker_atoms) == 0:
        raise ValueError("linker_atoms must contain at least one atom.")

    if residue_type not in permutation_residue_data:
        raise ValueError(f"No anchor atom indices are defined for "
                         f"residue type {residue_type}.")

    # ---------------------------------------------------------------
    # Get the anchor atom indices
    # ---------------------------------------------------------------

    anchor_indices = np.asarray(permutation_residue_data[residue_type], dtype=int)

    if np.any(anchor_indices < 0) or np.any(anchor_indices >= len(linker_atoms)):
        raise ValueError(f"At least one anchor atom index for residue type "
                         f"{residue_type} is outside the linker.")

    # ---------------------------------------------------------------
    # Copy the linker
    # ---------------------------------------------------------------

    wrapped_linker = linker_atoms.copy()

    # ---------------------------------------------------------------
    # Wrap all atoms relative to atom 0
    # ---------------------------------------------------------------

    positions = linker_atoms.get_positions()
    reference_position = positions[0]

    # Lattice parameters a, b, c
    Lx, Ly, Lz = linker_atoms.cell.lengths()

    cell_lengths = np.array([Lx, Ly, Lz])

    # Difference of every atom relative to atom 0
    displacement = positions - reference_position

    # Image coordinates
    wrapped_positions = positions - cell_lengths*np.round(displacement/cell_lengths)
    wrapped_linker.set_positions(wrapped_positions)

    # ---------------------------------------------------------------
    # Fit a plane through the anchor atoms
    # ---------------------------------------------------------------

    anchor_atoms = wrapped_linker[anchor_indices]
    center_of_anchor_atoms, vectors = SVD_plane_fit(anchor_atoms)

    if wrapped_to_coaa:
        wrapped_linker.set_positions(wrapped_linker.get_positions()-center_of_anchor_atoms)
    
    # Ensure the normal vector is pointing in the same direction as the reference vector
    vectors = normal_vector_check_and_corrections(vectors, wrapped_linker, anchor_indices[:3])

    return wrapped_linker, center_of_anchor_atoms, vectors





def optimize(atoms: Atoms, fmax: float = 0.1, steps: int = 200, hydrostatic_strain: bool = True, optimize_cell: bool = True) -> tuple[bool, int]:
    """
    Optimize the structure of the atoms object.

    Args:
        atoms (Atoms): Atoms object to be optimized
        fmax (float): Maximum force allowed on atoms. Default: 0.1
        steps (int): Maximum number of optimization steps. Default: 200
        hydrostatic_strain (bool): Whether to apply hydrostatic strain. Default: True
        optimize_cell (bool): Whether to optimize the cell. Default: True

    Returns:
        tuple[bool, int]: A tuple containing the convergence status and the number of optimization steps.
    """

    if optimize_cell:
        ucf = UnitCellFilter(atoms, hydrostatic_strain=hydrostatic_strain)
        opt = LBFGS(ucf, logfile="opt.log")
    else:
        opt = LBFGS(atoms, logfile="opt.log")

    opt.run(fmax=fmax, steps=steps)

    nOptimizationSteps = opt.nsteps

    return opt.converged(), nOptimizationSteps




def optimize_dftbp(dftb_current_optimization_log_file: Path | None)-> tuple[bool, int]:
    """
    Run DFTB+ optimization and check for convergence.
    Args:
        dftb_current_optimization_log_file (Path): Path to the temporary DFTB+ optimization log file. Default is "dftb.out".
    Returns:
        tuple[bool, int]: A tuple containing the convergence status and the number of optimization steps.
    """

    if dftb_current_optimization_log_file is None:
        dftb_current_optimization_log_file = Path("dftb.out")

    log_file = Path(dftb_current_optimization_log_file).resolve()
    working_directory = log_file.parent
    detailed_output = working_directory / "detailed.out"
    previous_detailed_output = working_directory / "previous_detailed.out"

    # ========== Run DFTB+ optimization ==========


    result = subprocess.run(
        ["dftb+",],
        cwd=working_directory,
        stdout=open(log_file, "w"),
        stderr=open("dftb.stderr", "w"),
        check=True,
    )
    
    # Update the snapshot only after DFTB+ has completed successfully, so it
    # always represents the most recently completed cycle.
    shutil.copyfile(detailed_output, previous_detailed_output)

    converged = False
    nsteps = None

    with open(detailed_output, "r") as f:
        for line in f:
            if "geometry converged" in line.lower():
                converged = True

            match = re.search(r"Geometry optimization step:\s*(\d+)", line)
            if match:
                nsteps = int(match.group(1))

    converged = converged and result.returncode == 0
    return converged, nsteps





def configured_residue_instances(atoms: Atoms,
                                 molecules: list[dict],
                                 residue_types) -> list[tuple[np.ndarray, int]]:
    """Return configured residue instances in their structure order."""
    molecule_counts = {
        molecule["residueNumber"]: molecule["atom_count"]
        for molecule in molecules
    }
    residue_numbers = atoms.get_array("residueNumber")
    instances = []

    for residue_type in residue_types or ():
        atom_count = molecule_counts[residue_type]
        indices = np.where(residue_numbers == residue_type)[0]
        residues = np.split(indices, np.arange(atom_count, len(indices), atom_count))
        if not all(len(residue) == atom_count for residue in residues):
            raise ValueError(
                f"Residue type {residue_type} does not contain complete molecules."
            )
        instances.extend((residue, residue_type) for residue in residues)

    instances.sort(key=lambda instance: instance[0][0])
    return instances


def residue_instance_index(atoms: Atoms,
                           molecules: list[dict],
                           residue_types,
                           global_indices: np.ndarray) -> int:
    """Return the log index of a selected configured residue instance."""
    selected_first_atom = int(global_indices[0])
    instances = configured_residue_instances(atoms, molecules, residue_types)
    for index, (indices, _) in enumerate(instances):
        if int(indices[0]) == selected_first_atom:
            return index
    raise ValueError("Selected residue is not configured for this operation.")


def configured_residue_instances(atoms: Atoms,
                                 molecules: list[dict],
                                 residue_types) -> list[tuple[np.ndarray, int]]:
    """Return configured residue instances in their structure order."""
    molecule_counts = {
        molecule["residueNumber"]: molecule["atom_count"]
        for molecule in molecules
    }
    residue_numbers = atoms.get_array("residueNumber")
    instances = []

    for residue_type in residue_types or ():
        atom_count = molecule_counts[residue_type]
        indices = np.where(residue_numbers == residue_type)[0]
        residues = np.split(indices, np.arange(atom_count, len(indices), atom_count))
        if not all(len(residue) == atom_count for residue in residues):
            raise ValueError(
                f"Residue type {residue_type} does not contain complete molecules."
            )
        instances.extend((residue, residue_type) for residue in residues)

    instances.sort(key=lambda instance: instance[0][0])
    return instances


def residue_instance_index(atoms: Atoms,
                           molecules: list[dict],
                           residue_types,
                           global_indices: np.ndarray) -> int:
    """Return the log index of a selected configured residue instance."""
    selected_first_atom = int(global_indices[0])
    instances = configured_residue_instances(atoms, molecules, residue_types)
    for index, (indices, _) in enumerate(instances):
        if int(indices[0]) == selected_first_atom:
            return index
    raise ValueError("Selected residue is not configured for this operation.")







def readGenFile(file_path: Path | None) -> Atoms:
    """Reads a GEN file and returns an ASE Atoms object."""
    from pathlib import Path

    if file_path is None:
        text = Path("dftb_in.template").read_text()
        match = re.search(r'OutputPrefix\s*=\s*"([^"]+)"', text)
        file_path = match.group(1) if match else None
        file_path = Path(file_path + ".gen")

    with open(Path(file_path), 'r') as f:
        lines = f.readlines()

    # Extract the number of atoms
    num_atoms = int(lines[0].strip().split()[0])

    # Extract the element symbols from the second line. The GEN atom
    # numbers are one-based indices into this file-defined mapping.
    atomic_symbols_by_number = lines[1].split()

    # Extract the atomic symbols and coordinates
    atomic_numbers = []
    atomic_symbols = []
    coordinates = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atomic_number = int(parts[1])
        atomic_numbers.append(atomic_number)
        if not 1 <= atomic_number <= len(atomic_symbols_by_number):
            raise ValueError(
                f"Atomic number {atomic_number} is not defined in the GEN "
                "file's element mapping."
            )
        atomic_symbols.append(atomic_symbols_by_number[atomic_number - 1])
        coordinates.append([float(coord) for coord in parts[2:5]])

    # Create an ASE Atoms object
    atoms = Atoms(symbols=atomic_symbols, positions=coordinates)

    lattice_vectors = np.array([
        [float(value) for value in line.split()]
        for line in lines[3 + num_atoms : 6 + num_atoms]
    ])

    atoms.set_cell(lattice_vectors)
    atoms.set_pbc([True, True, True])


    return atoms





def writeGenFile(atoms: Atoms) -> None:

    """Writes an ASE Atoms object to a GEN file."""
    
    text = Path("dftb_in.template").read_text()
    match = re.search(r'OutputPrefix\s*=\s*"([^"]+)"', text)
    GenFile = match.group(1) if match else None

    write(GenFile+".gen", atoms, format="gen")





def GenHSDfromTemplate(template_path: str, output_path: str, keyGUESS: str = "no") -> None:
    """
    Create a new HSD file from a template by replacing the __GUESS__ placeholder with the specified value.

    Parameters:
    - template_path (str): Path to the template HSD file.
    - output_path (str): Path where the new HSD file will be saved.
    - keyGUESS (str): The value to replace __GUESS__ with. Default is "no".

    Returns:
    - None
    """
    # Read the template HSD file
    with open(Path(template_path), 'r') as template_file:
        lines = template_file.readlines()

    # Find the line with __GUESS__ and replace it with yes/no
    line_index = next((i for i, line in enumerate(lines) if '__GUESS__' in line), None)
    if line_index is not None:
        lines[line_index] = lines[line_index].replace('__GUESS__', keyGUESS)
    # Write the new HSD file
    with open(Path(output_path), 'w') as output_file:
        output_file.writelines(lines)




def Energy_from_dftbp_log(log_file_path: Path = Path("detailed.out")) -> float:
    """
    Extracts the total energy from a DFTB+ log file.

    Args:
        log_file_path (Path): Path to the DFTB+ log file.

    Returns:
        float: Total energy in eV.

    Raises:
        ValueError: If the total energy is not found in the log file.
    """
    with open(log_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "Total energy" in line:
            energy = float(line.split()[-2])
            return energy

    raise ValueError("Total Energy not found in the log file.")




def get_output_prefix(file_path):
    """
    Extracts the value of OutputPrefix from a DFTB+ input file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "OutputPrefix" in line:
            output_prefix = line.split("=")[1].strip().strip('"')
            return output_prefix

    raise ValueError("OutputPrefix not found in the file.")





class PairRestraint(Calculator):
    """
    A custom ASE calculator that applies a restraint to pairs of atoms.
    Args:
        pairs: List of [atom_pair, r_max, k]
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, pairs, **kwargs):
        super().__init__(**kwargs)

        self.pairs = pairs

    def calculate(
        self,
        atoms=None,
        properties=implemented_properties,
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        forces = np.zeros((len(atoms), 3))
        energy = 0.0
        stress = np.zeros(6)

        for pair, r_max, k in self.pairs:
            #print("Pair:", pair)
            atom1, atom2 = pair

            vector = atoms.get_distance(atom1, atom2, mic=True, vector=True)
            distance = np.linalg.norm(vector)

            if distance > r_max and distance > 1e-5:

                delta = distance - r_max

                energy += 0.5 * k * delta**2

                force_vector = -(k * delta * vector / distance)

                forces[atom1] += force_vector
                forces[atom2] -= force_vector

                # Add stress contribution
                stress += np.array([
                    vector[0] * force_vector[0],
                    vector[1] * force_vector[1],
                    vector[2] * force_vector[2],
                    vector[1] * force_vector[2],
                    vector[0] * force_vector[2],
                    vector[0] * force_vector[1]])
        
        stress *= -1.0 / atoms.get_volume()
                

        self.results = {"energy": energy, "forces": forces, "stress": stress}







class MOF:
    """Stateful MOF Monte Carlo optimizer.

    Typical use::

        mof = MOF()
        mof.set_yaml("configuration.yaml")
        mof.set_moldescriptor("moldescriptor.dat")
        mof.thermocycle()

    ``set_yaml`` loads the structure and calculator; ``set_moldescriptor``
    assigns residue numbers needed by rotation and permutation moves.
    """

    def __init__(self):
        """Create an unconfigured optimizer; call the setup methods before use."""
        # Structure
        self.atoms = None
        self.moldescriptor = None
        self.rotational_log = None
        self.permutation_log = None
        self.dihedral_log = None

        # Input / output
        self.filename = None
        self.outputfile = None
        self.logfile = None
        self.trajectoryfile = None
        self.final_structurefile = None
        self.orientation_logfile = None
        self.permutation_logfile = None
        self.dihedral_logfile = None
        self.optimization_logfile = None
        self.dftbp_temporary_optimization_log_file = None

        # Calculation parameters
        self.job_type = None
        self.bits = None
        self.dtype = None
        self.dispersion = None
        self.hydrostatic_strain = None
        self.optimize_cell = None
        self.max_steps = None
        self.sampling_steps = None
        self.output_frequency = None
        self.version = None
        self.fmax = None
        self.T = None
        self.Tfirst = None
        self.Tinitial = None
        self.Tfinal = None
        self.Tsampling = None
        self.thermocycles = None
        self.cooling_first = None
        self.cooling_rate = None
        self.calculator = None
        self.constraints = None


        # Molecular operations
        self.permutational_residues = None
        self.rotational_residues = None
        self.angle_of_rotation = None
        self.weight_rotation = None
        self.weight_permutation = None
        self.dihedral_residues = None



        self.DEFAULT_CONFIGURATION_TEMPLATE = """# Configuration for MOF Monte Carlo optimization.
# Copy an input structure to filename before running the calculation.
filename: {input_filename}

# Plain XYZ trajectory log, readable by VMD. This file must not exist yet.
outputfile: trajectory.xyz

# Energy, temperature, and acceptance statistics. This file must not exist yet.
logfile: output.dat

# ASE binary trajectory. This file must not exist yet.
trajectoryfile: trajectory.traj

# Best structure after all thermocycles. This file must not exist yet.
final_structurefile: final_structure.xyz

# Optional operation logs. Set to null to disable them.
orientation_logfile: null
permutation_logfile: null
dihedral_logfile: null
optimization_logfile: null
# MACE model size: small, medium, or large.
version: medium

# Initial temperature for the first cycle, in K.
Tfirst: 5000.0

# Initial temperature for subsequent thermocycles, in K.
Tinit: 200.0

# Multiplicative cooling factor for the first cycle (0 < value < 1).
cooling_first: 0.995

# Final temperature for every cycle, in K.
Tfinal: 1.0

# Multiplicative cooling factor for subsequent cycles (0 < value <= 1).
cooling_rate: 0.99

# Number of thermocycles after the first cycle.
number_of_thermocycles: 1

# Relative frequencies of permutation and rotation moves.
weight_permutation: 1.0
weight_rotation: 1.0
weight_dihedral_flips: 1.0

# Maximum force for geometry optimization, in eV/Angstrom.
fmax: 0.1

# Maximum optimizer steps per trial move.
max_steps: 10000

# Operation-state log frequency. 0 disables these logs, negative values log
# after every successful Metropolis step, positive values log every Nth success.
output_frequency: -1

# Whether MACE includes dispersion interactions.
dispersion: true

# Whether only hydrostatic unit-cell strain is optimized.
hydrostatic_strain: true

# MACE numerical precision. Allowed values are 32 or 64.
bits: 32

# optimize cell parameters (bool)
optimize_cell_parameters: False

# Residue types eligible for rotation and local atom indices defining axes.
rotational_residues:
  3: [9, 12, 13, 14]

# Rotation angle in degrees.
angle_of_rotation: 180

# Residue types eligible for permutation and local anchor atom indices.
# At least two residue types are required when permutation is enabled.
permutational_residues:
  3: [9, 12, 13, 14]
  4: [8, 11, 12, 13]

# Residuenumbers and local atomic indices for dihedral flipping
# only necessary for non-planar linkers
# leave empty if no dihedral flipping is wanted
dihedral_residues:
 3: [[0,2,4,7,9,15], [8,11,12,13]]
"""

        self.DEFAULT_MOLDESCRIPTOR = """# Molecule definitions used to assign residue numbers.
# Format: molecule name, atom count, total charge; followed by atom rows.

# Moltype 1
Zn(II)        1       2
Zn            0       2.0

# Moltype 2
O(-II)        1      -2
O             0      -2.0

# Moltype 3
BDC-OH2-      17     -2
C             0       0.0
O             0       0.0
H             0       0.0
C             0       0.0
H             0       0.0
C             0       0.0
C             0       0.0
C             0       0.0
C             0       0.0
O             0       0.0
C             0       0.0
H             0       0.0
O             0       0.0
O             0       0.0
O             0       0.0
H             0       0.0
C             0       0.0

# Moltype 4
BDC2-         16     -2
C             0       0.0
H             0       0.0
C             0       0.0
H             0       0.0
C             0       0.0
C             0       0.0
C             0       0.0
C             0       0.0
O             0       0.0
C             0       0.0
H             0       0.0
O             0       0.0
O             0       0.0
O             0       0.0
H             0       0.0
C             0       0.0
"""


    def create_default_files(self,
                             directory: str | Path = ".",
                             input_filename: str = "initial.xyz",
                             configuration_filename: str = "configuration.yaml",
                             moldescriptor_filename: str = "moldescriptor.dat",
                             overwrite: bool = False) -> tuple[Path, Path]:
        """Create documented starter configuration and molecule-definition files.

        The generated files contain the current project's default MACE, annealing,
        residue-index, and molecule values. The input structure is not created;
        copy a PQ-style XYZ structure to ``input_filename`` before calling
        :meth:`MOF.set_yaml`.

        Args:
            directory: Directory in which both files are created.
            input_filename: Structure filename (xyz) written into the YAML file.
            configuration_filename: Name of the generated YAML configuration.
            moldescriptor_filename: Name of the generated molecule-definition file.
            overwrite: Replace existing target files when ``True``. Defaults to
                ``False`` to avoid accidental data loss.

        Returns:
            A tuple ``(configuration_path, moldescriptor_path)``.

        Raises:
            FileExistsError: If either target exists and ``overwrite`` is false.
            ValueError: If two target filenames resolve to the same path.
        """

        target_directory = Path(directory)
        target_directory.mkdir(parents=True, exist_ok=True)
        configuration_path = target_directory / configuration_filename
        moldescriptor_path = target_directory / moldescriptor_filename

        if configuration_path.resolve() == moldescriptor_path.resolve():
            raise ValueError("Configuration and moldescriptor filenames must differ.")

        targets = (configuration_path, moldescriptor_path)
        if not overwrite:
            existing = [path for path in targets if path.exists()]
            if existing:
                names = ", ".join(str(path) for path in existing)
                raise FileExistsError(f"Default file(s) already exist: {names}")

        configuration_path.write_text(
            self.DEFAULT_CONFIGURATION_TEMPLATE.format(input_filename=input_filename),
            encoding="utf-8",
        )
        moldescriptor_path.write_text(self.DEFAULT_MOLDESCRIPTOR, encoding="utf-8")

        return configuration_path, moldescriptor_path






    # Read the yaml 
    def set_yaml(self, filepath):
        """
        Checks the YAML file for its sanity and loads the parameters.
        Also loads the PQ data from the specified xyz file.

        Args:
            filepath (str): path to .yaml configuration file

        """

        paramsYaml = readYaml(filepath)

        self.job_type = paramsYaml["job_type"]

        self.filename = paramsYaml["filename"]


        self.outputfile = paramsYaml["outputfile"]
        self.logfile = paramsYaml["logfile"]
        self.trajectoryfile = paramsYaml["trajectoryfile"]
        self.final_structurefile = paramsYaml["final_structurefile"]
        self.orientation_logfile = paramsYaml["orientation_logfile"]
        self.permutation_logfile = paramsYaml["permutation_logfile"]
        self.dihedral_logfile = paramsYaml["dihedral_logfile"]
        self.dftbp_temporary_optimization_log_file = paramsYaml["dftbp_temporary_optimization_log_file"]

        self.version = paramsYaml["version"]

        self.bits = paramsYaml["bits"]

        self.dtype = paramsYaml["dtype"]

        self.dispersion = paramsYaml["dispersion"]

        self.hydrostatic_strain = paramsYaml["hydrostatic_strain"]

        self.optimize_cell = paramsYaml["optimize_cell_parameters"]

        self.optimization_logfile = paramsYaml["optimization_logfile"]

        self.output_frequency = paramsYaml["output_frequency"]

        self.fmax = float(paramsYaml["fmax"])

        self.permutational_residues = paramsYaml["permutational_residues"]

        self.rotational_residues = paramsYaml["rotational_residues"]

        self.dihedral_residues = paramsYaml["dihedral_residues"]

        self.angle_of_rotation = float(paramsYaml["angle_of_rotation"])

        self.weight_rotation = float(paramsYaml["weight_rotation"])

        self.weight_permutation = float(paramsYaml["weight_permutation"])

        self.weight_dihedral_flips = float(paramsYaml["weight_dihedral_flips"])
        
        self.max_steps = paramsYaml["max_steps"]

        if "annealing" in self.job_type:

            self.Tfirst = float(paramsYaml["Tfirst"])
            self.Tinitial = float(paramsYaml["Tinit"])

            self.Tfinal = float(paramsYaml["Tfinal"])

            self.thermocycles = int(paramsYaml["number_of_thermocycles"])

            self.cooling_first = float(paramsYaml["cooling_first"])
            self.cooling_rate = float(paramsYaml["cooling_rate"])

        else:
            self.Tsampling = float(paramsYaml["Tsampling"])

            self.sampling_steps = int(paramsYaml["sampling_steps"])

        if self.job_type == "annealing" or self.job_type == "sampling_constant_T":
            from mace.calculators import mace_mp

            self.atoms = pq_reader(self.filename)

            mace_calc = mace_mp(model = self.version, 
                                    device = "cuda", 
                                    default_dtype=self.dtype, 
                                    dispersion = self.dispersion)
            #print("\nSetting up MACE calculator")


        elif "dftbp" in self.job_type:
            self.atoms = readGenFile(None)

        else:
            raise ValueError(f"Unknown job_type: {self.job_type}")


        self.__set_moldescriptor__(paramsYaml["moldescriptor_file"])


        if "dftb" not in self.job_type:
            from ase.calculators.mixing import SumCalculator
            if paramsYaml["constraints"] is not None and paramsYaml["constraints"] != "None":
                self.constraints = paramsYaml["constraints"]
                
                pairs = []

                for residue_number, residue_constraints in self.constraints.items():
                    for local_pair, r_max, k in residue_constraints:
                        global_pairs = LocalToGlobalIndices(self.atoms, local_pair, residue_number, self.moldescriptor)

                        for global_pair in global_pairs:
                            pairs.append([global_pair, r_max, k])

                
                pair_calc = PairRestraint(pairs)

                self.calculator = SumCalculator([mace_calc, pair_calc])

                print("\nMace calculator with pair restraints is set up")
            else:
                self.calculator = mace_calc
            self.atoms.calc = self.calculator



    # read the moldescriptor
    def __set_moldescriptor__(self, filepath):
        """
        Load molecule definitions and assign residue numbers to ``self.atoms``.

        Args:
            filepath: Path to a moldescriptor file.

        Raises:
            RuntimeError: If no structure has been loaded with ``set_yaml``.
        """

        if self.atoms is None:
            raise RuntimeError("Call set_yaml before set_moldescriptor.")

        self.moldescriptor = read_moldescriptor(filepath)

        self.atoms = addResidueNumberToAtomsWithMoldescribtor(self.moldescriptor, self.atoms)

        rotating_instances = configured_residue_instances(
            self.atoms, self.moldescriptor, self.rotational_residues
        )
        permutating_instances = configured_residue_instances(
            self.atoms, self.moldescriptor, self.permutational_residues
        )
        dihedral_instances = configured_residue_instances(
            self.atoms, self.moldescriptor, self.dihedral_residues
        )
        
        self.rotational_log = np.zeros(len(rotating_instances), dtype=int)
        self.permutation_log = np.array(
            [residue_type for _, residue_type in permutating_instances], dtype=int
        )
        self.dihedral_log = np.zeros(len(dihedral_instances), dtype=int)


    # logging
    def LogXYZ(self):
        """Append the current structure to the configured plain XYZ log."""
        LogToXYZ(self.atoms, self.outputfile)





    def __extract_random_residue__(self):
        """
        Selects a random residue of one of the rotational residue types.

        Returns:
            -rotating_residue : ase.Atoms
                Randomly selected residue.
            -residue_number : int
                Residue type number.
            -global_indices : numpy array
                Array of global indices of the residue atoms
        """

        return extract_random_residue(self.atoms,
                               self.moldescriptor,
                               list(self.rotational_residues.keys()))




    def __extract_2_random_residues__(self):
        """
            Select two random residues of two different allowed permutational residue types.

            Returns:
                tuple
                    A tuple containing three lists:

                    ``residues`` : list of ase.Atoms
                        The two randomly selected residues.

                    ``residue_types`` : list of int
                        The residue type number corresponding to each selected
                        residue.

                    ``global_indices`` : list of numpy.ndarray
                        The global atom indices of each selected residue.

                    The entries in all three lists correspond to each other.
                    For example::

                        residues[0]
                        residue_types[0]
                        global_indices[0]

                    all describe the first selected residue.
            """
        
        return extract_2_random_residues(self.atoms,
                                         self.moldescriptor,
                                         self.permutational_residues)





    def rotate_random_residue(self) -> tuple[Atoms, int, int]:
        """
        Randomly select a residue from the rotational type (moldescriptor)
        and rotate it by the the angle specified in the yaml file.

        Returns:
            Ase.atoms : updated atoms object.
            int: rotation vector indicating the type of rotation applied.
        """

        residue, residue_number, global_indices = self.__extract_random_residue__()

        wrapped_residue, _, _ = preparation_routine(residue,
                                                    self.rotational_residues,
                                                    residue_number)

        rotated_residue, rot_axes = rotate_residue(wrapped_residue,
                                         residue_number,
                                         self.rotational_residues,
                                         self.angle_of_rotation)

        log_index = residue_instance_index(
            self.atoms, self.moldescriptor, self.rotational_residues, global_indices
        )
        return reImage1Residue(self.atoms, rotated_residue, global_indices), rot_axes, log_index


    def __extract_random_dihedral__(self):
        """Select a random residue eligible for dihedral flipping."""

        return extract_random_residue(self.atoms,
                                      self.moldescriptor,
                                      list(self.dihedral_residues.keys()))


    def flip_random_dihedral(self) -> tuple[Atoms, int]:
        """Mirror a random dihedral residue through its anchor plane."""

        residue, residue_number, global_indices = self.__extract_random_dihedral__()

        flipped_residue = Dihedral_Flipping(residue,
                                            residue_number,
                                            self.dihedral_residues)

        log_index = residue_instance_index(
            self.atoms, self.moldescriptor, self.dihedral_residues, global_indices
        )
        return reImage1Residue(self.atoms, flipped_residue, global_indices), log_index






    def swap_2_random_residues(self) -> tuple[Atoms, int, int, int, int]:
        """
        Randomly select 2 residues of different type and
        exchange their location. For this the anchor atoms
        specified in the yaml file are used to fit a plane
        and then orient them respectively.

        Returns:
            Ase.atoms : updated atoms object.
        """
    
        PermutatingResidues, PermutatingTypes, PermutatingGlobalIndices = self.__extract_2_random_residues__()
        
        permutating_residue_1, permutating_residue_2 = PermutatingResidues
        permutating_residue_type_1, permutating_residue_type_2 = PermutatingTypes
    
    
        # ========= calculate normal vectors and image residues==========
        wrapped_residue_1, coaa_1, vectors_1 = preparation_routine(permutating_residue_1,
                                                                    self.permutational_residues,
                                                                    permutating_residue_type_1)
    
        wrapped_residue_2, coaa_2, vectors_2 = preparation_routine(permutating_residue_2,
                                                                    self.permutational_residues,
                                                                    permutating_residue_type_2)
    
        # ======== swap the chosen residues ========
        swapped_residue_1, swapped_residue_2 = swap_residues([wrapped_residue_1, wrapped_residue_2],
                                                                [vectors_1, vectors_2],
                                                                [coaa_1, coaa_2])
    
        
        # ======== reimage the residues ==========    
        atoms = reImage2Residues(self.atoms,
                     [swapped_residue_1, swapped_residue_2],
                     PermutatingGlobalIndices)

        
        log_index_1 = residue_instance_index(
            self.atoms,
            self.moldescriptor,
            self.permutational_residues,
            PermutatingGlobalIndices[0],
        )
        log_index_2 = residue_instance_index(
            self.atoms,
            self.moldescriptor,
            self.permutational_residues,
            PermutatingGlobalIndices[1],
        )
        return (atoms, log_index_1, log_index_2,
                permutating_residue_type_1, permutating_residue_type_2)



    def __metropolis_step__(self, temperature: float) -> tuple[bool, str, dict, int]:
        """Perform one molecular move and return its result and operation.
        Returns:
            tuple: A tuple containing a boolean indicating if the move was accepted,
                   a string indicating the type of operation performed,
                   and metadata needed to update the corresponding operation log,
                    and the number of optimization steps taken.
        """

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        total_weight = (self.weight_permutation
                + self.weight_rotation
                + self.weight_dihedral_flips)
        if total_weight <= 0:
            raise ValueError("At least one molecular operation must be enabled")

        move_data = {}

        old_atoms = self.atoms.copy()
        old_energy = self.atoms.get_potential_energy()


        move_selector = np.random.rand() * total_weight
        if move_selector < self.weight_permutation:
            operation = "permutation"
            _, index_1, index_2, type_1, type_2 = self.swap_2_random_residues()
            move_data = {
                "index_1": index_1,
                "index_2": index_2,
                "type_1": type_1,
                "type_2": type_2,
            }
        elif move_selector < self.weight_permutation + self.weight_rotation:
            operation = "rotation"
            _, rotational_vector, linker_index = self.rotate_random_residue()
            move_data = {
                "rotational_vector": rotational_vector,
                "linker_index": linker_index,
            }
        else:
            operation = "dihedral"
            _, linker_index = self.flip_random_dihedral()
            move_data = {"linker_index": linker_index}

        converged, nOptimizationSteps = optimize(self.atoms,
                             fmax=self.fmax,
                             steps=self.max_steps,
                             hydrostatic_strain=self.hydrostatic_strain)
        if not converged:
            self.atoms = old_atoms
            #self.LogXYZ()
            self.atoms.calc = self.calculator
            return False, operation, move_data, nOptimizationSteps
        
        new_energy = self.atoms.get_potential_energy()
        acceptance_probability = min(1.0,
                                     np.exp(-(new_energy - old_energy) / (kB * temperature)))
        
        if np.random.rand() > acceptance_probability:
            self.atoms = old_atoms
            #self.LogXYZ()
            self.atoms.calc = self.calculator
            return False, operation, move_data, nOptimizationSteps

        
        return True, operation, move_data, nOptimizationSteps




    def __metropolis_step_dftbp__(self, temperature: float) -> tuple[bool, str, dict, int]:
        """Perform one molecular move and return its result and operation.
        Returns:
            tuple: A tuple containing a boolean indicating if the move was accepted,
                   a string indicating the type of operation performed,
                   and metadata needed to update the corresponding operation log,
                    and the number of optimization steps taken.
        """

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        total_weight = (self.weight_permutation
                + self.weight_rotation
                + self.weight_dihedral_flips)
        if total_weight <= 0:
            raise ValueError("At least one molecular operation must be enabled")

        move_data = {}

        old_atoms = self.atoms.copy()
        old_energy = Energy_from_dftbp_log()


        move_selector = np.random.rand() * total_weight
        if move_selector < self.weight_permutation:
            operation = "permutation"
            _, index_1, index_2, type_1, type_2 = self.swap_2_random_residues()
            move_data = {
                "index_1": index_1,
                "index_2": index_2,
                "type_1": type_1,
                "type_2": type_2,
            }
        elif move_selector < self.weight_permutation + self.weight_rotation:
            operation = "rotation"
            _, rotational_vector, linker_index = self.rotate_random_residue()
            move_data = {
                "rotational_vector": rotational_vector,
                "linker_index": linker_index,
            }
        else:
            operation = "dihedral"
            _, linker_index = self.flip_random_dihedral()
            move_data = {"linker_index": linker_index}

        # Write new coordinates to gen file for dftbp optimization
        writeGenFile(self.atoms)

        converged, nOptimizationSteps = optimize_dftbp(self.dftbp_temporary_optimization_log_file)
        if not converged:
            self.atoms = old_atoms
            #self.LogXYZ()
            return False, operation, move_data, nOptimizationSteps
        
        new_energy = Energy_from_dftbp_log()
        acceptance_probability = min(1.0,
                                     np.exp(-(new_energy - old_energy) / (kB * temperature)))
        
        if np.random.rand() > acceptance_probability:
            self.atoms = old_atoms
            #self.LogXYZ()
            return False, operation, move_data, nOptimizationSteps

        return True, operation, move_data, nOptimizationSteps





    def startSimulation(self)-> Atoms:
        """Start the simulation based on the job type specified in the configuration."""
        if self.job_type == "annealing":
            return self.thermocycle()
        elif self.job_type == "sampling_constant_T":
            return self.sampling_constant_T()
        elif self.job_type == "annealing_dftbp":
            return self.thermocycle_dftbp()
        elif self.job_type == "sampling_constant_T_dftbp":
            return self.sampling_constant_T_dftbp()
        else:
            raise ValueError(f"Unknown job type: {self.job_type}")



    def thermocycle(self,
                   logfile: str | None = None,
                   trajectory_name: str | None = None,
                   final_structure_name: str | None = None) -> Atoms:
        """Run the first high-temperature cycle followed by thermocycles.

        The first cycle starts at ``Tfirst`` and uses ``cooling_first``.
        Its resulting structure is then used as the input to each of the
        configured cycles starting at ``Tinitial``.

        Returns:
            ase.Atoms : The final structure after all thermocycles.
        """

        if self.atoms is None:
            raise ValueError("The structure must be loaded before thermocycle")
        if not 0 < self.cooling_first <= 1 or not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling rates must be greater than 0 and less or equal to 1")

        logfile = logfile or self.logfile or "output.dat"
        trajectory_name = trajectory_name or self.trajectoryfile or "traj_annealing.traj"
        final_structure_name = (
            final_structure_name or self.final_structurefile or "final_structure.xyz"
        )

        trajectory = Trajectory(trajectory_name, "a", self.atoms)




        def run_cycle(initial_temperature: float,
                      cooling_rate: float,
                      cycle_number: int,
                      log_file) -> Atoms:
            cycle_best = self.atoms.copy()
            cycle_best_energy = self.atoms.get_potential_energy()
            temperature = initial_temperature
            accepted = 0
            steps = 1
            operation_attempts = {"rotation": 0, "permutation": 0, "dihedral": 0}
            operation_accepts = {"rotation": 0, "permutation": 0, "dihedral": 0}

            log_file.write(
                f"Cycle #{cycle_number}\tSteps\tTime(s)\tEnergy(eV)\t\tTemperature(K)\tAcceptance Ratio"
                "\tRotation Attempts\tPermutation Attempts"
                "\tDihedral Attempts\tRotation Acceptance Ratio"
                "\tPermutation Acceptance Ratio\tDihedral Acceptance Ratio\n"
            )
            log_file.write(
                f"\t\t{time() - start_time:.3f}\t{cycle_best_energy:.5f}\tInitial structure!"
                "\t0\t0\t0.0000\t0.0000\n"
            )
            log_file.flush()

            trajectory.write(self.atoms)
            LogToXYZ(self.atoms, self.outputfile)

            while temperature > self.Tfinal:
                accepted_move, operation, move_data, nOptimizationSteps = self.__metropolis_step__(temperature)
                operation_attempts[operation] += 1
                current_energy = self.atoms.get_potential_energy()
                if accepted_move:
                    operation_accepts[operation] += 1
                    accepted += int(accepted_move)

                    trajectory.write(self.atoms)
                    LogToXYZ(self.atoms, self.outputfile)
                    LogOptimizationSteps(self.optimization_logfile, steps, nOptimizationSteps, operation[0])

                    match operation:
                        case "rotation":
                            update_rotation(self.rotational_log,move_data["rotational_vector"],move_data["linker_index"])

                        case "permutation":
                            update_permutation(self.permutation_log,move_data["index_1"],move_data["index_2"])

                        case "dihedral":
                            update_dihedral(self.dihedral_log,move_data["linker_index"])

                    if self.output_frequency < 0:
                        append_array_log(self.orientation_logfile, self.rotational_log, steps)
                        append_array_log(self.permutation_logfile, self.permutation_log, steps)
                        append_array_log(self.dihedral_logfile, self.dihedral_log, steps)

                        write_acceptance_ratio(log_file,
                                               steps,
                                                time() - start_time,
                                                current_energy,
                                                temperature,
                                                accepted / steps,
                                                operation_attempts["rotation"],
                                                operation_attempts["permutation"],
                                                operation_attempts["dihedral"],
                                                operation_accepts["rotation"] / operation_attempts["rotation"] if operation_attempts["rotation"] else 0.0,
                                                operation_accepts["permutation"] / operation_attempts["permutation"] if operation_attempts["permutation"] else 0.0,
                                                operation_accepts["dihedral"] / operation_attempts["dihedral"] if operation_attempts["dihedral"] else 0.0)
                
    

                if ((self.output_frequency > 0) and (steps % self.output_frequency == 0)):
                    append_array_log(self.orientation_logfile, self.rotational_log, steps)
                    append_array_log(self.permutation_logfile, self.permutation_log, steps)
                    append_array_log(self.dihedral_logfile, self.dihedral_log, steps)

                    write_acceptance_ratio(log_file,
                                        steps,
                                        time() - start_time,
                                        current_energy,
                                        temperature,
                                        accepted / steps,
                                        operation_attempts["rotation"],
                                        operation_attempts["permutation"],
                                        operation_attempts["dihedral"],
                                        operation_accepts["rotation"] / operation_attempts["rotation"] if operation_attempts["rotation"] else 0.0,
                                        operation_accepts["permutation"] / operation_attempts["permutation"] if operation_attempts["permutation"] else 0.0,
                                        operation_accepts["dihedral"] / operation_attempts["dihedral"] if operation_attempts["dihedral"] else 0.0)

                if current_energy < cycle_best_energy:
                    cycle_best = self.atoms.copy()
                    cycle_best_energy = current_energy
                    LogToXYZ(cycle_best, "BestCandidate.xyz", "w")
                    


                temperature *= cooling_rate
                steps += 1

            print(f"Cycle #{cycle_number} completed after {time() - start_time:.2f} seconds and {steps} steps.\n",
                  f"Best energy: {cycle_best_energy:.2f} eV.\n")

            cycle_best.calc = self.atoms.calc
            return cycle_best

        print("\n Setup complete. Initial energy is {:.2f} eV.\n".format(self.atoms.get_potential_energy()),
              "Starting thermocycle with {} cycles\n".format(self.thermocycles),
              "This may take a while, get a coffee and relax!\n")

        with open(logfile, "a") as log_file:
            self.atoms = run_cycle(self.Tfirst, self.cooling_first, 0, log_file)

            for cycle_number in range(1, self.thermocycles + 1):
                self.atoms = run_cycle(self.Tinitial,
                                       self.cooling_rate,
                                       cycle_number,
                                       log_file)

        self.atoms.calc = self.calculator
        #self.atoms.get_potential_energy()
        write(final_structure_name, self.atoms, format="extxyz", append=True)
        trajectory.close()

        return self.atoms




    def thermocycle_dftbp(self,
                   logfile: str | None = None,
                   trajectory_name: str | None = None,
                   final_structure_name: str | None = None) -> Atoms:
        """Run the first high-temperature cycle followed by thermocycles.

        The first cycle starts at ``Tfirst`` and uses ``cooling_first``.
        Its resulting structure is then used as the input to each of the
        configured cycles starting at ``Tinitial``.

        Returns:
            ase.Atoms : The final structure after all thermocycles.
        """

        if self.atoms is None:
            raise ValueError("The structure must be loaded before thermocycle")
        if not 0 < self.cooling_first <= 1 or not 0 < self.cooling_rate <= 1:
            raise ValueError("cooling rates must be greater than 0 and less or equal to 1")

        logfile = logfile or self.logfile or "output.dat"
        trajectory_name = trajectory_name or self.trajectoryfile or "traj_annealing.traj"
        final_structure_name = (
            final_structure_name or self.final_structurefile or "final_structure.xyz"
        )

        trajectory = Trajectory(trajectory_name, "a", self.atoms)



        # ============= change/create hsd file from template ==========================
        # No initial charges for the first run, as they will be guessed by dftb+.
        GenHSDfromTemplate("dftb_in.template", "dftb_in.hsd", keyGUESS = "no")
        




        def run_cycle(initial_temperature: float,
                      cooling_rate: float,
                      cycle_number: int,
                      log_file) -> Atoms:
            cycle_best = self.atoms.copy()
            cycle_best_energy = Energy_from_dftbp_log()
            temperature = initial_temperature
            accepted = 0
            steps = 1
            operation_attempts = {"rotation": 0, "permutation": 0, "dihedral": 0}
            operation_accepts = {"rotation": 0, "permutation": 0, "dihedral": 0}

            log_file.write(
                f"Cycle #{cycle_number}\tSteps\tTime(s)\tEnergy(eV)\t\tTemperature(K)\tAcceptance Ratio"
                "\tRotation Attempts\tPermutation Attempts"
                "\tDihedral Attempts\tRotation Acceptance Ratio"
                "\tPermutation Acceptance Ratio\tDihedral Acceptance Ratio\n"
            )
            log_file.write(
                f"\t\t{time() - start_time:.3f}\t{cycle_best_energy:.2f}\tInitial structure!"
                "\t0\t0\t0.0000\t0.0000\n"
            )
            log_file.flush()

            trajectory.write(self.atoms)
            LogToXYZ(self.atoms, self.outputfile)


            # Preoptimize the initial structure with DFTB+ to guess the charges
            optimize_dftbp(self.dftbp_temporary_optimization_log_file)

            # from now on read the charges from the previous optimization step
            GenHSDfromTemplate("dftb_in.template", "dftb_in.hsd", keyGUESS = "yes")


            while temperature > self.Tfinal:
                accepted_move, operation, move_data, nOptimizationSteps = self.__metropolis_step_dftbp__(temperature)
                operation_attempts[operation] += 1
                current_energy = Energy_from_dftbp_log()
                if accepted_move:
                    operation_accepts[operation] += 1
                    accepted += int(accepted_move)

                    self.atoms = readGenFile(Path("geoopt_lattice.gen"))
                    self.atoms = addResidueNumberToAtomsWithMoldescribtor(self.moldescriptor,
                                                                          self.atoms)

                    operation_accepts[operation] += 1

                    trajectory.write(self.atoms)
                    LogToXYZ(self.atoms, self.outputfile)
                    LogOptimizationSteps(self.optimization_logfile, steps, nOptimizationSteps, operation[0])

                    match operation:
                        case "rotation":
                            update_rotation(self.rotational_log,move_data["rotational_vector"],move_data["linker_index"])

                        case "permutation":
                            update_permutation(self.permutation_log,move_data["index_1"],move_data["index_2"])

                        case "dihedral":
                            update_dihedral(self.dihedral_log,move_data["linker_index"])

                    if self.output_frequency < 0:
                        append_array_log(self.orientation_logfile, self.rotational_log, steps)
                        append_array_log(self.permutation_logfile, self.permutation_log, steps)
                        append_array_log(self.dihedral_logfile, self.dihedral_log, steps)

                        write_acceptance_ratio(log_file,
                                               steps,
                                                time() - start_time,
                                                current_energy,
                                                temperature,
                                                accepted / steps,
                                                operation_attempts["rotation"],
                                                operation_attempts["permutation"],
                                                operation_attempts["dihedral"],
                                                operation_accepts["rotation"] / operation_attempts["rotation"] if operation_attempts["rotation"] else 0.0,
                                                operation_accepts["permutation"] / operation_attempts["permutation"] if operation_attempts["permutation"] else 0.0,
                                                operation_accepts["dihedral"] / operation_attempts["dihedral"] if operation_attempts["dihedral"] else 0.0)
                
                
                if ((self.output_frequency > 0) and (steps % self.output_frequency == 0)):
                    append_array_log(self.orientation_logfile, self.rotational_log, steps)
                    append_array_log(self.permutation_logfile, self.permutation_log, steps)
                    append_array_log(self.dihedral_logfile, self.dihedral_log, steps)

                    write_acceptance_ratio(log_file,
                                        steps,
                                        time() - start_time,
                                        current_energy,
                                        temperature,
                                        accepted / steps,
                                        operation_attempts["rotation"],
                                        operation_attempts["permutation"],
                                        operation_attempts["dihedral"],
                                        operation_accepts["rotation"] / operation_attempts["rotation"] if operation_attempts["rotation"] else 0.0,
                                        operation_accepts["permutation"] / operation_attempts["permutation"] if operation_attempts["permutation"] else 0.0,
                                        operation_accepts["dihedral"] / operation_attempts["dihedral"] if operation_attempts["dihedral"] else 0.0)

                if current_energy < cycle_best_energy:
                    cycle_best = self.atoms.copy()
                    cycle_best_energy = current_energy
                    LogToXYZ(cycle_best, "BestCandidate.xyz", "w")


                temperature *= cooling_rate
                steps += 1

            print(f"Cycle #{cycle_number} completed after {time() - start_time:.2f} seconds and {steps} steps.\n",
                  f"Best energy: {cycle_best_energy:.2f} eV.\n")

            cycle_best.calc = self.atoms.calc
            return cycle_best

        print("\n Setup complete. Initial energy is {:.2f} eV.\n".format(Energy_from_dftbp_log()),
              "Starting thermocycle with {} cycles\n".format(self.thermocycles),
              "This may take a while, get a coffee and relax!\n")

        with open(logfile, "a") as log_file:
            self.atoms = run_cycle(self.Tfirst, self.cooling_first, 0, log_file)

            for cycle_number in range(1, self.thermocycles + 1):
                self.atoms = run_cycle(self.Tinitial,
                                       self.cooling_rate,
                                       cycle_number,
                                       log_file)

        write(final_structure_name, self.atoms, format="extxyz", append=True)
        trajectory.close()

        return self.atoms







    def sampling_constant_T(self) -> Atoms:
        """Run a constant-temperature sampling simulation.

        Returns:
            ase.Atoms : The final structure after all sampling steps.
        """

        if self.atoms is None:
            raise ValueError("The structure must be loaded before sampling")

        cycle_best = self.atoms.copy()
        cycle_number = 0
        cycle_best_energy = self.atoms.get_potential_energy()
        accepted = 0
        operation_attempts = {"rotation": 0, "permutation": 0, "dihedral": 0}
        operation_accepts = {"rotation": 0, "permutation": 0, "dihedral": 0}
        trajectory = Trajectory(self.trajectoryfile, "a", self.atoms)
        

        print("\n Setup complete. Initial energy is {:.2f} eV.\n".format(self.atoms.get_potential_energy()),
                "Starting sampling with {} steps at {} K\n".format(self.sampling_steps, self.Tsampling),
                "This may take a while, get a coffee and relax!\n")

        

        with open(self.logfile, "a") as log_file:
            log_file.write(
                f"Cycle #{cycle_number}\tSteps\tTime(s)\tEnergy(eV)\t\tTemperature(K)\tAcceptance Ratio"
                "\tRotation Attempts\tPermutation Attempts"
                "\tDihedral Attempts\tRotation Acceptance Ratio"
                "\tPermutation Acceptance Ratio\tDihedral Acceptance Ratio\n"
            )
            log_file.write(
                f"\t\t{time() - start_time:.3f}\t{cycle_best_energy:.2f}\tInitial structure!"
                "\t0\t0\t0.0000\t0.0000\n"
            )
            log_file.flush()

        trajectory.write(self.atoms)
        LogToXYZ(self.atoms, self.outputfile)



        with open(self.logfile, "a") as log_file:

            for step in range(self.sampling_steps):
                accepted_move, operation, move_data, nOptimizationSteps = self.__metropolis_step__(self.Tsampling)
                operation_attempts[operation] += 1
                current_energy = self.atoms.get_potential_energy()
                if accepted_move:
                    operation_accepts[operation] += 1
                    accepted += int(accepted_move)

                    LogOptimizationSteps(self.optimization_logfile, step, nOptimizationSteps, operation[0])

                    match operation:
                        case "rotation":
                            update_rotation(self.rotational_log,move_data["rotational_vector"],move_data["linker_index"])

                        case "permutation":
                            update_permutation(self.permutation_log,move_data["index_1"],move_data["index_2"])

                        case "dihedral":
                            update_dihedral(self.dihedral_log,move_data["linker_index"])

                    LogToXYZ(self.atoms, self.outputfile)

                    if self.output_frequency < 0:
                        append_array_log(self.orientation_logfile, self.rotational_log, step)
                        append_array_log(self.permutation_logfile, self.permutation_log, step)
                        append_array_log(self.dihedral_logfile, self.dihedral_log, step)

                        write_acceptance_ratio(log_file,
                                                step,
                                                time() - start_time,
                                                current_energy,
                                                self.Tsampling,
                                                accepted / (step+1),
                                                operation_attempts["rotation"],
                                                operation_attempts["permutation"],
                                                operation_attempts["dihedral"],
                                                operation_accepts["rotation"] / operation_attempts["rotation"] if operation_attempts["rotation"] else 0.0,
                                                operation_accepts["permutation"] / operation_attempts["permutation"] if operation_attempts["permutation"] else 0.0,
                                                operation_accepts["dihedral"] / operation_attempts["dihedral"] if operation_attempts["dihedral"] else 0.0)
                
                

                if ((self.output_frequency > 0) and (step % self.output_frequency == 0)):
                    append_array_log(self.orientation_logfile, self.rotational_log, step)
                    append_array_log(self.permutation_logfile, self.permutation_log, step)
                    append_array_log(self.dihedral_logfile, self.dihedral_log, step)

                    write_acceptance_ratio(log_file,
                                        step,
                                        time() - start_time,
                                            current_energy,
                                            self.Tsampling,
                                            accepted / (step+1),
                                            operation_attempts["rotation"],
                                            operation_attempts["permutation"],
                                            operation_attempts["dihedral"],
                                            operation_accepts["rotation"] / operation_attempts["rotation"] if operation_attempts["rotation"] else 0.0,
                                            operation_accepts["permutation"] / operation_attempts["permutation"] if operation_attempts["permutation"] else 0.0,
                                            operation_accepts["dihedral"] / operation_attempts["dihedral"] if operation_attempts["dihedral"] else 0.0)

                if current_energy < cycle_best_energy:
                    cycle_best = self.atoms.copy()
                    cycle_best_energy = current_energy
                    LogToXYZ(cycle_best, "BestCandidate.xyz", "w")




        self.atoms.calc = self.calculator
        write(self.final_structurefile, cycle_best, format="extxyz", append=True)
        trajectory.close()


        self.atoms.calc = self.calculator
        return self.atoms






    def sampling_constant_T_dftbp(self) -> Atoms:
        """Run a constant-temperature sampling simulation using DFTB+.

        Returns:
            ase.Atoms : The final structure after all sampling steps.
        """

        print("\n Starting DFTB+ sampling with {} steps at {} K\n".format(self.sampling_steps, self.Tsampling),
              "Make sure you want to use DFTB+ for this simulation, as it may take a while!\n")

        if self.atoms is None:
            raise ValueError("The structure must be loaded before sampling")

        cycle_best = self.atoms.copy()
        cycle_number = 0
        cycle_best_energy = Energy_from_dftbp_log()
        accepted = 0
        operation_attempts = {"rotation": 0, "permutation": 0, "dihedral": 0}
        operation_accepts = {"rotation": 0, "permutation": 0, "dihedral": 0}
        trajectory = Trajectory(self.trajectoryfile, "a", self.atoms)
        

        print("\n Setup complete. Initial energy is {:.2f} eV.\n".format(Energy_from_dftbp_log()),
                "Starting sampling with {} steps at {} K\n".format(self.sampling_steps, self.Tsampling),
                "This may take a while, get a coffee and relax!\n")

        

        with open(self.logfile, "a") as log_file:
            log_file.write(
                f"Cycle #{cycle_number}\tSteps\tTime(s)\tEnergy(eV)\t\tTemperature(K)\tAcceptance Ratio"
                "\tRotation Attempts\tPermutation Attempts"
                "\tDihedral Attempts\tRotation Acceptance Ratio"
                "\tPermutation Acceptance Ratio\tDihedral Acceptance Ratio\n"
            )
            log_file.write(
                f"\t\t{time() - start_time:.3f}\t{cycle_best_energy:.2f}\tInitial structure!"
                "\t0\t0\t0.0000\t0.0000\n"
            )
            log_file.flush()

        trajectory.write(self.atoms)
        LogToXYZ(self.atoms, self.outputfile)


        # Preoptimize the initial structure with DFTB+ to guess the charges
        GenHSDfromTemplate("dftb_in.template", "dftb_in.hsd", keyGUESS = "no")
        optimize_dftbp(self.dftbp_temporary_optimization_log_file)

        # from now on read the charges from the previous optimization step
        GenHSDfromTemplate("dftb_in.template", "dftb_in.hsd", keyGUESS = "yes")



        with open(self.logfile, "a") as log_file:

            for step in range(self.sampling_steps):
                accepted_move, operation, move_data, nOptimizationSteps = self.__metropolis_step_dftbp__(self.Tsampling)
                operation_attempts[operation] += 1
                current_energy = Energy_from_dftbp_log()
                if accepted_move:
                    operation_accepts[operation] += 1
                    accepted += int(accepted_move)

                    self.atoms = readGenFile(Path("geoopt_lattice.gen"))
                    self.atoms = addResidueNumberToAtomsWithMoldescribtor(self.moldescriptor,
                                                                          self.atoms)

                    LogOptimizationSteps(self.optimization_logfile, step, nOptimizationSteps, operation[0])

                    match operation:
                        case "rotation":
                            update_rotation(self.rotational_log,move_data["rotational_vector"],move_data["linker_index"])

                        case "permutation":
                            update_permutation(self.permutation_log,move_data["index_1"],move_data["index_2"])

                        case "dihedral":
                            update_dihedral(self.dihedral_log,move_data["linker_index"])

                    LogToXYZ(self.atoms, self.outputfile)

                    if self.output_frequency < 0:
                        append_array_log(self.orientation_logfile, self.rotational_log, step)
                        append_array_log(self.permutation_logfile, self.permutation_log, step)
                        append_array_log(self.dihedral_logfile, self.dihedral_log, step)

                        write_acceptance_ratio(log_file,
                                        step,
                                        time() - start_time,
                                            current_energy,
                                            self.Tsampling,
                                            accepted / (step+1),
                                            operation_attempts["rotation"],
                                            operation_attempts["permutation"],
                                            operation_attempts["dihedral"],
                                            operation_accepts["rotation"] / operation_attempts["rotation"] if operation_attempts["rotation"] else 0.0,
                                            operation_accepts["permutation"] / operation_attempts["permutation"] if operation_attempts["permutation"] else 0.0,
                                            operation_accepts["dihedral"] / operation_attempts["dihedral"] if operation_attempts["dihedral"] else 0.0)

                if ((self.output_frequency > 0) and (step % self.output_frequency == 0)):
                    append_array_log(self.orientation_logfile, self.rotational_log, step)
                    append_array_log(self.permutation_logfile, self.permutation_log, step)
                    append_array_log(self.dihedral_logfile, self.dihedral_log, step)

                    write_acceptance_ratio(log_file,
                                        step,
                                        time() - start_time,
                                            current_energy,
                                            self.Tsampling,
                                            accepted / (step+1),
                                            operation_attempts["rotation"],
                                            operation_attempts["permutation"],
                                            operation_attempts["dihedral"],
                                            operation_accepts["rotation"] / operation_attempts["rotation"] if operation_attempts["rotation"] else 0.0,
                                            operation_accepts["permutation"] / operation_attempts["permutation"] if operation_attempts["permutation"] else 0.0,
                                            operation_accepts["dihedral"] / operation_attempts["dihedral"] if operation_attempts["dihedral"] else 0.0)

                if current_energy < cycle_best_energy:
                    cycle_best = self.atoms.copy()
                    cycle_best_energy = current_energy
                    LogToXYZ(cycle_best, "BestCandidate.xyz", "w")

        write(self.final_structurefile, cycle_best, format="extxyz", append=True)
        trajectory.close()

        return self.atoms



def read_output_log(log_filename: str | Path,
                    dihedral_log_file: str | Path | None = None,
                    rotation_log_file: str | Path | None = None,
                    permutation_log_file: str | Path | None = None,
                    xyz_log_file: str | Path | None = None,
                    optimization_steps_log_file: str | Path | None = None,
                    include_initial_values: bool = False,
                    initial_dihedral_log: list | None = None,
                    initial_rotation_log: list | None = None,
                    initial_permutation_log: list | None = None)-> pd.DataFrame:
    """Read simulation records and optionally attach array snapshots by step."""

    log_filename = Path(log_filename)

    def read_array_log(log_file: str | Path) -> list[np.ndarray]:
        snapshots = []
        with Path(log_file).open("r", encoding="utf-8") as array_file:
            for line in array_file:
                fields = line.split()
                if len(fields) < 2:
                    continue
                try:
                    values = np.asarray([int(value) for value in fields[1:]], dtype=int)
                except ValueError:
                    continue
                snapshots.append(values)
        return snapshots


    steps = []
    times = []
    energies = []
    temperatures = []
    acceptance_ratios = []
    rotation_attempts = []
    permutation_attempts = []
    dihedral_attempts = []
    rotation_acceptance_ratios = []
    permutation_acceptance_ratios = []
    dihedral_acceptance_ratios = []
    cycles = []
    xyz_snapshots = []
    cycle = None

    with log_filename.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            stripped_line = line.strip()

            if stripped_line.startswith("Cycle #"):
                cycle = int(stripped_line.split("#", 1)[1].split("\t", 1)[0])
                continue

            fields = stripped_line.split()
            if len(fields) not in (7, 9, 11) or cycle is None:
                continue

            try:
                values = list(map(float, fields))
            except ValueError:
                continue

            step, time, energy, temperature, acceptance_ratio = values[:5]
            if len(values) == 11:
                operation_values = values[5:]
            elif len(values) == 9:
                operation_values = values[5:] + [np.nan, np.nan]
            else:
                operation_values = [np.nan] * 6


            steps.append(step)
            times.append(time)
            energies.append(energy)
            temperatures.append(temperature)
            acceptance_ratios.append(acceptance_ratio)
            rotation_attempts.append(operation_values[0])
            permutation_attempts.append(operation_values[1])
            dihedral_attempts.append(operation_values[2])
            rotation_acceptance_ratios.append(operation_values[3])
            permutation_acceptance_ratios.append(operation_values[4])
            dihedral_acceptance_ratios.append(operation_values[5])
            cycles.append(cycle)


    if not energies:
        raise ValueError(f"No simulation records found in '{log_filename}'.")

    out = pd.DataFrame({
        "steps": np.asarray(steps, dtype=int),
        "times": np.asarray(times),
        "energies": np.asarray(energies),
            "temperatures": np.asarray(temperatures),
            "acceptance_ratios": np.asarray(acceptance_ratios),
            "rotation_attempts": np.asarray(rotation_attempts, dtype=int),
            "permutation_attempts": np.asarray(permutation_attempts, dtype=int),
            "dihedral_attempts": np.asarray(dihedral_attempts, dtype=int),
            "rotation_acceptance_ratios": np.asarray(rotation_acceptance_ratios),
            "permutation_acceptance_ratios": np.asarray(permutation_acceptance_ratios),
            "dihedral_acceptance_ratios": np.asarray(dihedral_acceptance_ratios),
            "cycles": np.asarray(cycles, dtype=int)})

    if xyz_log_file is not None:
        if optimization_steps_log_file is None:
            raise ValueError("Optimization steps log file has to be provided if xyz_log_file is provided.")

        accepted_changes_MC_steps = np.loadtxt(
            optimization_steps_log_file,
            delimiter="\t",
            usecols=0,
            skiprows=1,
            dtype=int,
        )
        accepted_changes_MC_steps = np.atleast_1d(accepted_changes_MC_steps)

        # Find indices where sequence resets
        split_indices = np.where(np.diff(accepted_changes_MC_steps) < 0)[0] + 1

        # Split the array into sub-arrays based on the identified indices
        accepted_changes_MC_steps_by_cycle = np.split(accepted_changes_MC_steps, split_indices)
  
        # analyze acceptance ratio
        cycle_with_accepted_moves = {
            int(cycle_number): np.any(
                cycle_data["acceptance_ratios"].to_numpy() > 0.0
            )
            for cycle_number, cycle_data in out.groupby("cycles", sort=False)
        }

        cycle_numbers = [int(cycle_number) for cycle_number in out["cycles"].unique()]
        accepted_steps_by_cycle = {
            cycle_number: np.array([], dtype=int)
            for cycle_number in cycle_numbers
        }
        
        cycles_with_accepted_moves = [
            cycle_number
            for cycle_number in cycle_numbers
            if cycle_with_accepted_moves.get(cycle_number, True)
        ]
        cycles_with_accepted_moves = set(cycles_with_accepted_moves)
        
        if len(accepted_changes_MC_steps_by_cycle) > len(cycles_with_accepted_moves):
            raise ValueError(
                "The optimization-step log contains more cycle segments "
                "than the output log contains energy-changing cycles."
            )
        accepted_steps_by_cycle.update(
            zip(cycles_with_accepted_moves, accepted_changes_MC_steps_by_cycle)
        )
        with Path(xyz_log_file).open("r", encoding="utf-8") as xyz_file:
            def next_frame() -> Atoms:
                frame = _read_pq_frame(xyz_file)
                if frame is None:
                    raise EOFError(
                        "The PQ trajectory contains fewer frames than the "
                        "output and optimization logs require."
                    )
                return frame

            previous_snapshot = None
            previous_cycle = None

            for row_index, row in out.iterrows():
                cycle_number = int(row["cycles"])
                step = int(row["steps"])

                if cycle_number != previous_cycle:
                    if cycle_number not in accepted_steps_by_cycle:
                        raise ValueError(
                            f"No accepted-step data exists for cycle "
                            f"{cycle_number}."
                        )

                    if (
                        previous_snapshot is None
                        or cycle_number in cycles_with_accepted_moves
                    ):
                        previous_snapshot = next_frame()
                    previous_cycle = cycle_number

                if step in accepted_steps_by_cycle[cycle_number]:
                    accepted_snapshot = next_frame()
                    previous_snapshot = accepted_snapshot

                xyz_snapshots.append(previous_snapshot)

        out["atoms"] = pd.Series(xyz_snapshots, index=out.index, dtype=object)




    for column_name, array_log_file in (
        ("dihedral_log", dihedral_log_file),
        ("rotation_log", rotation_log_file),
        ("permutation_log", permutation_log_file),
    ):
        if array_log_file is None:
            continue

        snapshots = read_array_log(array_log_file)
        out[column_name] = pd.Series(
            [
                snapshots[index] if index < len(snapshots) else None
                for index in range(len(out))
            ],
            dtype=object,
        )



    # Initial values
    if include_initial_values:
        with log_filename.open("r", encoding="utf-8") as log_file:
            n = 0
            for index, line in enumerate(log_file):
                stripped_line = line.strip()
                if "Initial structure!" in stripped_line:
                    InitStep = 0
                    InitTime = float(stripped_line.split("\t")[0])
                    InitEnergy = float(stripped_line.split("\t")[1])

                    next_line = next(log_file).strip()
                    InitTemperature = float(next_line.split("\t")[3])
                    
                    InitAcceptanceRatio = 0.0

                    InitRotationAttempts = 0
                    InitRotationAcceptanceRatio = 0.0

                    InitPermutationAttempts = 0
                    InitPermutationAcceptanceRatio = 0.0

                
                    InitDihedralAttempts = 0
                    InitDihedralAcceptanceRatio = 0.0
                    
                    InitCycle = n

                    InitDihedralLog = [np.nan]
                    InitRotationLog = [np.nan]
                    InitPermutationLog = [np.nan]

                    if n == 0:
                        if dihedral_log_file is not None:
                            InitDihedralLog = np.asarray(initial_dihedral_log, dtype=int)
                        if rotation_log_file is not None:
                            InitRotationLog = np.asarray(initial_rotation_log, dtype=int)
                        if permutation_log_file is not None:
                            InitPermutationLog = np.asarray(initial_permutation_log, dtype=int)

                    else:
                        if dihedral_log_file is not None:
                            InitDihedralLog = np.asarray(out["dihedral_log"].iloc[index-2], dtype=int)
                        if rotation_log_file is not None:
                            InitRotationLog = np.asarray(out["rotation_log"].iloc[index-2], dtype=int)
                        if permutation_log_file is not None:
                            InitPermutationLog = np.asarray(out["permutation_log"].iloc[index-2], dtype=int)
                    n += 1

                    if dihedral_log_file is not None and rotation_log_file is not None and permutation_log_file is not None:
                        outInitial = [InitStep, InitTime, InitEnergy, InitTemperature, InitAcceptanceRatio,
                                    InitRotationAttempts, InitPermutationAttempts, InitDihedralAttempts,
                                    InitRotationAcceptanceRatio, InitPermutationAcceptanceRatio, InitDihedralAcceptanceRatio,
                                    InitCycle,
                                    InitDihedralLog, InitRotationLog, InitPermutationLog]
                    elif dihedral_log_file is not None and rotation_log_file is not None:
                        outInitial = [InitStep, InitTime, InitEnergy, InitTemperature, InitAcceptanceRatio,
                                    InitRotationAttempts, InitPermutationAttempts, InitDihedralAttempts,
                                    InitRotationAcceptanceRatio, InitPermutationAcceptanceRatio, InitDihedralAcceptanceRatio,
                                    InitCycle,
                                    InitDihedralLog, InitRotationLog]
                    elif dihedral_log_file is not None and permutation_log_file is not None:
                        outInitial = [InitStep, InitTime, InitEnergy, InitTemperature, InitAcceptanceRatio,
                                    InitRotationAttempts, InitPermutationAttempts, InitDihedralAttempts,
                                    InitRotationAcceptanceRatio, InitPermutationAcceptanceRatio, InitDihedralAcceptanceRatio,
                                    InitCycle,
                                    InitDihedralLog, InitPermutationLog]
                    elif rotation_log_file is not None and permutation_log_file is not None:
                        outInitial = [InitStep, InitTime, InitEnergy, InitTemperature, InitAcceptanceRatio,
                                    InitRotationAttempts, InitPermutationAttempts, InitDihedralAttempts,
                                    InitRotationAcceptanceRatio, InitPermutationAcceptanceRatio, InitDihedralAcceptanceRatio,
                                    InitCycle,
                                    InitRotationLog, InitPermutationLog]
                    elif dihedral_log_file is not None:
                        outInitial = [InitStep, InitTime, InitEnergy, InitTemperature, InitAcceptanceRatio,
                                    InitRotationAttempts, InitPermutationAttempts, InitDihedralAttempts,
                                    InitRotationAcceptanceRatio, InitPermutationAcceptanceRatio, InitDihedralAcceptanceRatio,
                                    InitCycle,
                                    InitDihedralLog]
                    elif rotation_log_file is not None:
                        outInitial = [InitStep, InitTime, InitEnergy, InitTemperature, InitAcceptanceRatio,
                                    InitRotationAttempts, InitPermutationAttempts, InitDihedralAttempts,
                                    InitRotationAcceptanceRatio, InitPermutationAcceptanceRatio, InitDihedralAcceptanceRatio,
                                    InitCycle,
                                    InitRotationLog]
                    elif permutation_log_file is not None:
                        outInitial = [InitStep, InitTime, InitEnergy, InitTemperature, InitAcceptanceRatio,
                                    InitRotationAttempts, InitPermutationAttempts, InitDihedralAttempts,
                                    InitRotationAcceptanceRatio, InitPermutationAcceptanceRatio, InitDihedralAcceptanceRatio,
                                    InitCycle,
                                    InitPermutationLog]
                    else:
                        outInitial = [InitStep, InitTime, InitEnergy, InitTemperature, InitAcceptanceRatio,
                                    InitRotationAttempts, InitPermutationAttempts, InitDihedralAttempts,
                                    InitRotationAcceptanceRatio, InitPermutationAcceptanceRatio, InitDihedralAcceptanceRatio,
                                    InitCycle]

                    if xyz_log_file is not None:
                        if len(out["atoms"]) == 0:
                            raise ValueError(
                                "Cannot add initial values because the XYZ "
                                "trajectory contains no frames."
                            )
                        # The atoms column is added before the array-log
                        # columns, so it must follow InitCycle here as well.
                        outInitial.insert(12, out["atoms"].iloc[0])

                    out.loc[index-1.5] = pd.Series(outInitial, index=out.columns, dtype=object)
                    out = out.sort_index().reset_index(drop=True)

    return out





def get_final_structure_logs(xyz_log_file: str | Path,
                             optimization_steps_log_file: str | Path,
                             xyz_final_structure: str | Path,
                             log_filename: str | Path | None = None,
                             dihedral_log_file: str | Path | None = None,
                             rotation_log_file: str | Path | None = None,
                             permutation_log_file: str | Path | None = None,
                             initial_dihedral_log: list | None = None,
                             initial_rotation_log: list | None = None,
                             initial_permutation_log: list | None = None) -> pd.DataFrame:
    """Get the final structure and logs from a simulation run.

    Args:
        xyz_log_file (str or Path): The path to the XYZ trajectory file.
        xyz_final_structure (str or Path): The path to the final structure file.
        log_filename (str or Path): The path to the main log file.
        dihedral_log_file (str or Path, optional): The path to the dihedral log file.
        rotation_log_file (str or Path, optional): The path to the rotation log file.
        permutation_log_file (str or Path, optional): The path to the permutation log file.
        initial_dihedral_log (list, optional): The initial dihedral log.
        initial_rotation_log (list, optional): The initial rotation log.
        initial_permutation_log (list, optional): The initial permutation log.
        xyz_log_file (str or Path, optional): The path to the XYZ trajectory file.

    Returns:
        pd.DataFrame: A DataFrame containing the final structure and logs.
    """

    from ase.geometry import cellpar_to_cell



    if initial_dihedral_log is not None or \
         initial_rotation_log is not None or \
         initial_permutation_log is not None:
        include_initial_values = True
    else:
        include_initial_values = False

    df = read_output_log(log_filename,
                         dihedral_log_file=dihedral_log_file,
                         rotation_log_file=rotation_log_file,
                         permutation_log_file=permutation_log_file,
                         xyz_log_file=xyz_log_file,
                         optimization_steps_log_file=optimization_steps_log_file,
                         include_initial_values=include_initial_values,
                         initial_dihedral_log=initial_dihedral_log,
                         initial_rotation_log=initial_rotation_log,
                         initial_permutation_log=initial_permutation_log)
    
    if df.empty:
        raise ValueError(f"No data found in '{log_filename}'.")

    final_structure = pq_reader(xyz_final_structure)
    if final_structure is None:
        raise ValueError(f"Could not read final structure from '{xyz_final_structure}'.")

    # Match lattice parameter precission of final_structure with the one in the logs
    atoms_ref = df["atoms"][0]
    cellpar_ref = atoms_ref.cell.cellpar()
    cellpar_final = final_structure.cell.cellpar()
    precision = len(str(cellpar_ref[0]).split(".")[1])
    cellpar_final_truncated = (np.trunc(cellpar_final * 10**precision) / 10**precision)
    final_structure.set_cell(cellpar_to_cell(cellpar_final_truncated),scale_atoms=False)
    
    index_of_final_structure = df.index[df["atoms"] == final_structure].tolist()
    if len(index_of_final_structure) > 1:
        index_of_final_structure = index_of_final_structure[0]  # Take the first occurrence if multiple matches are found

    if not index_of_final_structure:
        raise ValueError(f"Final structure not found in the data.")

    return df.iloc[index_of_final_structure]  # Return the row(s) with the final structure/logs






def create_default_files(directory: str | Path = ".",
                         input_filename: str = "initial.xyz",
                         configuration_filename: str = "configuration.yaml",
                         moldescriptor_filename: str = "moldescriptor.dat",
                         overwrite: bool = False) -> tuple[Path, Path]:
    """Create starter YAML and moldescriptor files without configuring a MOF.

    Example::

        from MOFOperations import create_default_files

        configuration, moldescriptor = create_default_files("my_run")

    See :meth:`MOF.create_default_files` for the full argument and safety
    documentation. Existing files are not overwritten unless ``overwrite`` is
    explicitly set to ``True``.
    """

    return MOF().create_default_files(
        directory=directory,
        input_filename=input_filename,
        configuration_filename=configuration_filename,
        moldescriptor_filename=moldescriptor_filename,
        overwrite=overwrite,
    )


    











if __name__ == "__main__":
    import sys

    if sys.argv[1] == "--help" or sys.argv[1] == "-H":
        print("\nUsage: python MOFOperations.py <configuration.yaml>")
        print("Run the MOFOperations module with a YAML configuration file.")
        print("Make sure to load mace first.")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("\nUsage: python MOFOperations.py <configuration.yaml>")
        sys.exit(1)

    if not Path(sys.argv[1]).is_file():
        print(f"\nError: '{sys.argv[1]}' is not a valid file.")
        sys.exit(1)

    mof = MOF()
    mof.set_yaml(Path(sys.argv[1]))
    mof.startSimulation()




    
