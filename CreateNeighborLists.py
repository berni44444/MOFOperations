from genericpath import exists
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from MOFOperations import *
from ase.geometry import find_mic
from time import time


NEIGHBOR_CATEGORIES = ("in_line", "orthogonal")


def _create_neighbor_list_for_dataset_item(task):
    """Worker for generating one neighbor list outside the dataset loop."""
    (
        atoms_key,
        atoms,
        moldescriptor_file,
        orientation_indices_dict,
        cutoff_distance,
        angle_tolerance,
        sort_up_down,
    ) = task
    current_neighbor_list = create_neighbor_list(
        atoms,
        moldescriptor_file,
        orientation_indices_dict=orientation_indices_dict,
        cutoff_distance=cutoff_distance,
        angle_tolerance=angle_tolerance,
        sort_up_down=sort_up_down,
    )
    return atoms_key, current_neighbor_list


def _prepare_linker_neighbor_list(
    neighbor_list: dict[int, dict[str, np.ndarray]],
    rotating_residue: set[int],
) -> tuple[
    list[tuple[int, list[tuple[int, int]], bool]],
    dict[int, int],
    bool,
]:
    """Normalize neighbor entries and collect metadata reused per data row."""
    linker_residues = {}
    rotating_linker_numbers = set()
    prepared_rows = []
    include_in_line2 = False

    for neighbors in neighbor_list.values():
        for entries in neighbors.values():
            entries = np.asarray(entries)
            if entries.ndim != 2 or entries.shape[0] < 3:
                continue
            for linker_number, residue_number in entries[1:3].T:
                linker_number = int(linker_number)
                residue_number = int(residue_number)
                linker_residues[linker_number] = residue_number
                if residue_number in rotating_residue:
                    rotating_linker_numbers.add(linker_number)

    rotating_positions = {
        linker_number: position
        for position, linker_number in enumerate(sorted(rotating_linker_numbers))
    }

    for reference_index, neighbors in neighbor_list.items():
        reference_index = int(reference_index)
        reference_residue = linker_residues.get(reference_index)
        if reference_residue is None:
            raise ValueError(
                f"Residue number for reference linker {reference_index} "
                "is missing from the neighbor list."
            )

        normalized = {}
        for category in NEIGHBOR_CATEGORIES:
            entries = np.asarray(neighbors[category])
            entries = (
                entries.T
                if entries.ndim == 2 and entries.shape[0] == 3
                else entries
            )
            normalized[category] = entries

        orthogonal = normalized["orthogonal"]
        in_line = normalized["in_line"]
        if len(orthogonal) != 8:
            raise ValueError(
                f"Reference linker {reference_index} has "
                f"{len(orthogonal)} orthogonal neighbors; expected 8."
            )
        if len(in_line) not in (1, 2):
            raise ValueError(
                f"Reference linker {reference_index} has {len(in_line)} "
                "in-line neighbors; expected 1 or 2."
            )

        if len(in_line) == 2:
            include_in_line2 = True
        slots = [
            (reference_index, reference_residue),
            *(
                (int(entry[1]), int(entry[2]))
                for entry in in_line
            ),
            *(
                (int(entry[1]), int(entry[2]))
                for entry in orthogonal
            ),
        ]
        prepared_rows.append((reference_index, slots, len(in_line) == 2))

    return prepared_rows, rotating_positions, include_in_line2


def create_linker_dataset(
    data: pd.DataFrame,
    neighbor_list: dict[int, dict[str, np.ndarray]] | None = None,
    rotating_residue: list[int] = [4],
    dihedral_column: str = "dihedral_log",
    rotation_column: str = "rotation_log",
    gen_neighbor_list: bool = False,
    atoms_column: str = "atoms",
    moldescriptor_file: str | Path | None = None,
    orientation_indices_dict: dict[int, list[int]] | None = None,
    cutoff_distance: float | None = 14.0,
    angle_tolerance: float = 15.0,
    sort_up_down: bool = True,
    show_progress: bool = True,
    parallel_neighbor_lists: bool = True,
    neighbor_list_workers: int | None = 6,
) -> pd.DataFrame:
    """Create one encoded neighbor row for every linker configuration.

    Rotating linkers are encoded as ``drr``: ``d`` is the dihedral state and
    ``rr`` is the two-bit rotation state. Linkers of other residue types are
    encoded as ``-1``. The neighbor-list columns are ``ref``, ``in_line1``,
    optionally ``in_line2``, and the orthogonal neighbors.

    The array logs contain only rotating linkers. Their entries are matched to
    rotating linkers in ascending absolute linker-number order for each
    generated neighbor list. Neighbor-list metadata is normalized once and
    generated lists are reused when the same ``Atoms`` object appears again.
    When ``gen_neighbor_list`` is true, all unique neighbor lists are generated
    before dataset encoding. Set ``parallel_neighbor_lists`` to true to use
    separate worker processes for that precomputation.
    """
    if dihedral_column not in data or rotation_column not in data:
        raise KeyError(
            f"BrData must contain '{dihedral_column}' and '{rotation_column}'."
        )

    if gen_neighbor_list:
        if atoms_column not in data:
            raise KeyError(
                f"gen_neighbor_list=True requires an '{atoms_column}' column "
                "containing ASE Atoms objects."
            )
        if moldescriptor_file is None:
            raise ValueError(
                "moldescriptor_file is required when gen_neighbor_list=True."
            )
    elif neighbor_list is None:
        raise ValueError(
            "Provide neighbor_list or set gen_neighbor_list=True."
        )

    if neighbor_list is not None and not gen_neighbor_list:
        warnings.warn(
            "Neighbor list is provided. Ensure that it matches the data. "
            "If permutations are applied, the neighbor list may not be valid."
        )

    dataset_rows = []
    include_in_line2 = False
    prepared_neighbor_list_cache = {}
    generated_neighbor_list_cache = {}
    rotating_residue_set = set(rotating_residue)
    column_positions = {
        column: position for position, column in enumerate(data.columns)
    }
    if not gen_neighbor_list:
        prepared_neighbor_list_cache[id(neighbor_list)] = (
            _prepare_linker_neighbor_list(neighbor_list, rotating_residue_set)
        )
    else:
        effective_orientation_indices = orientation_indices_dict or {
            4: [5, 6],
            3: [5, 6],
        }
        unique_atoms = {}
        for data_row in data.itertuples(index=False, name=None):
            atoms = data_row[column_positions[atoms_column]]
            unique_atoms.setdefault(id(atoms), atoms)

        tasks = [
            (
                atoms_key,
                atoms,
                moldescriptor_file,
                effective_orientation_indices,
                cutoff_distance,
                angle_tolerance,
                sort_up_down,
            )
            for atoms_key, atoms in unique_atoms.items()
        ]

        if parallel_neighbor_lists and len(tasks) > 1:
            with ProcessPoolExecutor(
                max_workers=neighbor_list_workers
            ) as executor:
                generated_neighbor_list_cache.update(
                    tqdm(
                        executor.map(
                            _create_neighbor_list_for_dataset_item,
                            tasks,
                        ),
                        total=len(tasks),
                        desc="Creating neighbor lists",
                        unit="structure",
                        disable=not show_progress,
                    )
                )
        else:
            for task in tqdm(
                tasks,
                desc="Creating neighbor lists",
                unit="structure",
                disable=not show_progress,
            ):
                atoms_key, current_neighbor_list = (
                    _create_neighbor_list_for_dataset_item(task)
                )
                generated_neighbor_list_cache[atoms_key] = (
                    current_neighbor_list
                )

    for data_row in tqdm(
        data.itertuples(index=False, name=None),
        total=len(data),
        desc="Creating linker dataset",
        unit="row",
        disable=not show_progress,
    ):
        atoms = data_row[column_positions[atoms_column]] if gen_neighbor_list else None
        if gen_neighbor_list:
            atoms_key = id(atoms)
            current_neighbor_list = generated_neighbor_list_cache[atoms_key]
        else:
            current_neighbor_list = neighbor_list

        cache_key = id(atoms) if gen_neighbor_list else id(neighbor_list)
        prepared_rows, rotating_positions, has_in_line2 = (
            prepared_neighbor_list_cache.get(cache_key)
            or _prepare_linker_neighbor_list(
                current_neighbor_list, rotating_residue_set
            )
        )
        prepared_neighbor_list_cache[cache_key] = (
            prepared_rows,
            rotating_positions,
            has_in_line2,
        )
        dihedral_values = np.asarray(
            data_row[column_positions[dihedral_column]]
        )
        rotation_values = np.asarray(
            data_row[column_positions[rotation_column]]
        )

        def encode_linker(linker_number: int, residue_number: int) -> str:
            if residue_number not in rotating_residue_set:
                return "-1"
            position = rotating_positions.get(linker_number)
            if position is None:
                raise ValueError(
                    f"Rotating linker {linker_number} is missing from the "
                    "neighbor list."
                )
            dihedral = int(dihedral_values[position])
            rotation = int(rotation_values[position])
            if dihedral not in (0, 1) or rotation not in range(4):
                raise ValueError(
                    f"Invalid rotating-linker state ({dihedral}, {rotation}) "
                    f"for linker {linker_number}."
                )
            return f"{dihedral}{rotation:02b}"

        if has_in_line2 and not include_in_line2:
            include_in_line2 = True
            dataset_rows = [
                values[:2] + ["-1"] + values[2:]
                for values in dataset_rows
            ]

        for reference_index, slots, has_second_in_line in prepared_rows:
            encoded_row = [
                encode_linker(*slots[0]),
                encode_linker(*slots[1]),
            ]
            if include_in_line2:
                encoded_row.append(
                    encode_linker(*slots[2])
                    if has_second_in_line
                    else "-1"
                )
            encoded_row.extend(
                encode_linker(*slot)
                for slot in slots[3 if has_second_in_line else 2:]
            )
            dataset_rows.append(encoded_row)

    columns = ["ref", "in_line1"]
    if include_in_line2:
        columns.append("in_line2")
    columns.extend(f"o{index}" for index in range(1, 9))

    return pd.DataFrame(dataset_rows, columns=columns)


def save_neighbor_list(
    neighbor_list: dict[int, dict[str, np.ndarray]],
    filename: str | Path,
) -> None:
    """Save a neighbor list in a compact format that supports variable sizes."""
    filename = Path(filename)
    linker_indices = np.array(sorted(neighbor_list), dtype=np.int64)
    arrays = {"linker_indices": linker_indices}

    for category in NEIGHBOR_CATEGORIES:
        entries = [
            np.asarray(neighbor_list[index][category], dtype=np.int64).T
            for index in linker_indices
        ]
        lengths = np.array([len(entry) for entry in entries], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        values = (
            np.concatenate(entries, axis=0)
            if offsets[-1] > 0
            else np.empty((0, 3), dtype=np.int64)
        )

        arrays[f"{category}_values"] = values
        arrays[f"{category}_offsets"] = offsets

    np.savez_compressed(filename, **arrays)


def load_neighbor_list(
    filename: str | Path,
) -> dict[int, dict[str, np.ndarray]]:
    """Load a neighbor list saved by :func:`save_neighbor_list`."""
    filename = Path(filename)

    with np.load(filename) as data:
        linker_indices = data["linker_indices"]
        neighbor_list = {}

        for position, linker_index in enumerate(linker_indices):
            categories = {}
            for category in NEIGHBOR_CATEGORIES:
                offsets = data[f"{category}_offsets"]
                start, end = offsets[position:position + 2]
                values = data[f"{category}_values"][start:end]
                categories[category] = values.T.copy()

            neighbor_list[int(linker_index)] = categories

    return neighbor_list




def count_linkers(atoms: Atoms, moldescriptor: list[dict]):
    """Count the number of individual linkers in the given atoms based on the moldescriptor."""
    # Residue numbers corresponding to linkers (more than one atom)
    linker_residue_numbers = [
        molecule["residueNumber"]
        for molecule in moldescriptor
        if molecule["atom_count"] > 1
    ]

    # Count all atoms belonging to linkers
    residue_numbers = atoms.get_array("residueNumber")

    
    # Count the number of individual linkers of each type
    number_of_linkers = {}
    for resNum in residue_numbers:

        number_of_linkers_with_resNum = sum(
            np.sum(residue_numbers == resNum) // molecule["atom_count"]
            for molecule in moldescriptor
            if molecule["atom_count"] > 1 and molecule["residueNumber"] == resNum)

        number_of_linkers[resNum] = number_of_linkers_with_resNum

    

    return {x:y for x,y in number_of_linkers.items() if y > 0}








def unwrap_atoms(atoms: Atoms, moldescriptor: list[dict]) -> Atoms:
    """Unwrap each linker independently, preserving atom order and arrays.

    ``Atoms.wrap()`` stores only the wrapped coordinates, so the original
    periodic image of an atom cannot be recovered directly.  Reconstruct the
    linker image by growing it from one atom and placing each next atom at its
    nearest periodic image relative to an atom that has already been placed.
    """
    unwrapped_atoms = atoms.copy()
    wrapped_positions = unwrapped_atoms.get_positions()
    positions = wrapped_positions.copy()
    residue_numbers = unwrapped_atoms.get_array("residueNumber")

    for molecule in moldescriptor:
        atom_count = molecule["atom_count"]
        residue_number = molecule["residueNumber"]

        if atom_count <= 1:
            continue

        indices = np.flatnonzero(residue_numbers == residue_number)

        if len(indices) % atom_count != 0:
            raise ValueError(
                f"Unexpected atom count for residue {residue_number}"
            )

        for start in range(0, len(indices), atom_count):
            linker_indices = indices[start:start + atom_count]
            placed = np.zeros(atom_count, dtype=bool)
            placed[0] = True
            positions[linker_indices[0]] = wrapped_positions[linker_indices[0]]

            while not np.all(placed):
                best_distance = np.inf
                best_anchor = None
                best_displacement = None

                for atom_position, atom_index in enumerate(linker_indices):
                    if placed[atom_position]:
                        continue

                    for anchor_position, anchor_index in enumerate(linker_indices):
                        if not placed[anchor_position]:
                            continue

                        displacement, distance = find_mic(
                            wrapped_positions[atom_index]
                            - wrapped_positions[anchor_index],
                            cell=unwrapped_atoms.cell,
                            pbc=unwrapped_atoms.pbc,
                        )

                        if distance < best_distance:
                            best_distance = distance
                            best_atom = atom_position
                            best_anchor = anchor_index
                            best_displacement = displacement

                positions[linker_indices[best_atom]] = (
                    positions[best_anchor] + best_displacement
                )
                placed[best_atom] = True

    unwrapped_atoms.set_positions(positions)
    return unwrapped_atoms



def calculate_linker_orientation(linker: Atoms, orientation_indices: list):

    vector = np.array(linker[orientation_indices[1]].position) - np.array(linker[orientation_indices[0]].position)
    orientation = vector / np.linalg.norm(vector)
    return orientation




def create_neighbor_list(
    atoms,
    moldescriptor_file,
    orientation_indices_dict = {4: [5,6], 3:[5,6]},
    cutoff_distance: float | None = None,
    output_file: str | None = None,
    angle_tolerance: float = 15.0,
    mode: str = "w",
    sort_up_down: bool = True
) -> dict[int, dict[str, np.ndarray]]:
    """Return neighboring linkers for every linker.

    The returned dictionary is keyed by a sequential linker index. Each value
    contains ``(3, N)`` integer arrays for ``"in_line"`` and
    ``"orthogonal"``. Row zero contains the global index of the first atom of
    each neighboring linker, row one contains its absolute zero-based linker
    number, and row two contains its residue number.
    """


    moldescriptor = read_moldescriptor(Path(moldescriptor_file))

    atoms = addResidueNumberToAtomsWithMoldescribtor(moldescriptor, atoms)

    atoms = unwrap_atoms(atoms, moldescriptor)

    linker_residue_numbers = count_linkers(atoms, moldescriptor)

    if cutoff_distance is None or cutoff_distance <= 0:
        cutoff_distance = atoms.get_cell().max() * 0.4

    linker_types = list(linker_residue_numbers)
    linkers_by_type = {
        residue_number: atoms[
            atoms.get_array("residueNumber") == residue_number
        ]
        for residue_number in linker_types
    }
    global_indices_by_type = {
        residue_number: np.flatnonzero(
            atoms.get_array("residueNumber") == residue_number
        )
        for residue_number in linker_types
    }
    
    linker_records = []
    for residue_number in linker_types:
        residue_length = next(molecule["atom_count"] for molecule in moldescriptor if molecule["residueNumber"] == residue_number)

        for linker_index in range(linker_residue_numbers[residue_number]):
            start_index = linker_index * residue_length
            linker = linkers_by_type[residue_number][start_index:start_index + residue_length]
            carbon_indices = [index for index, atom in enumerate(linker) if atom.symbol == "C"]
            com = linker[carbon_indices].get_center_of_mass()
            vector = calculate_linker_orientation(linker, orientation_indices_dict[residue_number])
            global_index = global_indices_by_type[residue_number][start_index]

            linker_records.append(
                (global_index, len(linker_records), residue_number, com, vector)
            )

    global_indices = np.asarray([record[0] for record in linker_records])
    linker_numbers = np.asarray([record[1] for record in linker_records])
    residue_numbers = np.asarray([record[2] for record in linker_records])
    coms = np.asarray([record[3] for record in linker_records], dtype=float)
    orientations = np.asarray(
        [record[4] for record in linker_records], dtype=float
    )

    # Calculate all minimum-image COM displacements once instead of once for
    # every reference/neighbor pair.
    pair_displacements, pair_distances = find_mic(
        (coms[None, :, :] - coms[:, None, :]).reshape(-1, 3),
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    pair_displacements = pair_displacements.reshape(
        len(linker_records), len(linker_records), 3
    )
    pair_distances = pair_distances.reshape(
        len(linker_records), len(linker_records)
    )

    neighbor_list = {}
    for reference_index, reference_global_index in enumerate(global_indices):
        reference_vector = orientations[reference_index]
        distances = pair_distances[reference_index]
        displacements = pair_displacements[reference_index]
        candidate_mask = (
            (distances <= cutoff_distance)
            & (np.arange(len(linker_records)) != reference_index)
        )

        orientation_alignment = np.clip(
            np.abs(orientations @ reference_vector), -1.0, 1.0
        )
        orientation_angles = np.rad2deg(np.arccos(orientation_alignment))

        com_directions = np.zeros_like(displacements)
        nonzero_distances = distances > 0
        np.divide(
            displacements,
            distances[:, None],
            out=com_directions,
            where=nonzero_distances[:, None],
        )
        com_alignment = np.clip(
            np.abs(com_directions @ reference_vector), -1.0, 1.0
        )
        com_angles = np.rad2deg(np.arccos(com_alignment))

        in_line_mask = (
            candidate_mask
            & (orientation_angles <= angle_tolerance)
            & (com_angles <= angle_tolerance)
        )
        orthogonal_mask = (
            candidate_mask
            & ~in_line_mask
            & (np.abs(orientation_angles - 90) <= angle_tolerance)
        )

        neighbor_indices = {}
        neighbor_directions = {}
        for category, category_mask in (
            ("in_line", in_line_mask),
            ("orthogonal", orthogonal_mask),
        ):
            indices = np.flatnonzero(category_mask)
            directions = displacements[indices] @ reference_vector
            if sort_up_down:
                order = np.argsort(directions < 0, kind="stable")
                indices = indices[order]
                directions = directions[order]
            neighbor_indices[category] = [
                (
                    global_indices[index],
                    linker_numbers[index],
                    residue_numbers[index],
                )
                for index in indices
            ]
            neighbor_directions[category] = directions

        in_line_count = len(neighbor_indices["in_line"])
        orthogonal_count = len(neighbor_indices["orthogonal"])
        if in_line_count != 1 or orthogonal_count != 8:
            warnings.warn(
                f"Linker {reference_index} (first atom index "
                f"{reference_global_index}) has {in_line_count} in-line "
                f"and {orthogonal_count} orthogonal neighbors; expected "
                "1 in-line and 8 orthogonal neighbors. Consider choosing "
                "different neighbor-list parameters.",
                UserWarning,
                stacklevel=2,
            )

        neighbor_list[reference_index] = {
            category: np.asarray(entries, dtype=int).T
            if entries
            else np.empty((3, 0), dtype=int)
            for category, entries in neighbor_indices.items()
        }
    if output_file is not None:
        output_path = Path(output_file)
        if output_path.suffix.lower() == ".npz":
            if mode != "w":
                raise ValueError("Appending to an NPZ neighbor list is not supported.")
            save_neighbor_list(neighbor_list, output_path)
        else:
            with output_path.open("w" if mode == "w" else "a", encoding="utf-8") as output:
                for linker_index, neighbors in neighbor_list.items():
                    output.write(f"Linker {linker_index}:\n")
                    for category, entries in neighbors.items():
                        output.write(f"  {category}:\n")
                        for global_index, linker_number, residue_number in entries.T:
                            output.write(
                                f"    Residue Index: {global_index}, "
                                f"Linker Number: {linker_number}, "
                                f"Residue Number: {residue_number}\n"
                            )

        print(f"Neighbor list written to '{output_path}'.")

    return neighbor_list



if __name__ == "__main__":
    import sys
    from pathlib import Path

    if sys.argv[1] == "--help" or sys.argv[1] == "-H" or len(sys.argv) < 2:
        print("Usage: python CreateNeighborLists.py <yaml_file>"
              "\n"
              "  <yaml_file>: Path to the YAML file containing the configuration parameters:\n"
              "     <atoms_file>: Path to the XYZ file containing the atomic structure.\n"
              "     <moldescriptor_file>: Path to the moldescriptor file.\n"
              "     <orientation_indices_dict>: Optional. A dictionary mapping residue numbers to lists of two atom indices for orientation calculation. Default is {4: [5,6], 3:[5,6]}.\n"
              "     <output_file>: Optional. Path to the output file where the neighbor list will be saved. Default is None (no output file).\n"
              "     <output_mode>: Optional. Mode for writing the output file. Default is 'w' (write). Use 'a' to append to an existing file.\n"
              "     <cutoff_distance>: Optional. Cutoff distance for neighbor search. Default is None (will be set to 40% of the maximum cell dimension).\n")
        sys.exit(0)


    paramsYaml = Path(sys.argv[1])
    if not paramsYaml.is_file():
        print(f"Error: YAML file '{paramsYaml}' does not exist.")
        sys.exit(1)

    


    # Load the parameters from the YAML file.
    with open(paramsYaml, "r") as f:
        params = yaml.safe_load(f)

    # Extract the parameters.
    atoms_file = params.get("atoms_file", None)
    moldescriptor_file = params.get("moldescriptor_file", None)
    orientation_indices_dict = params.get("orientation_indices_dict", {4: [5,6], 3:[5,6]})
    output_file = params.get("output_file", None)
    cutoff_distance = params.get("cutoff_distance", None)
    output_mode = params.get("output_mode", "w")

    if output_file is not None:
        output_file = Path(output_file)
    if exists(output_file) and output_mode == "w":
        print(f"Warning: Output file '{output_file}' already exists and will be overwritten.\n"
              "Confirm with 'y' to proceed or 'n' to cancel.")
        user_input = input("Proceed? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Operation cancelled by user.")
            sys.exit(0)

    start_time = time()


    atoms = pq_reader(atoms_file)
    neighbor_list = create_neighbor_list(
        atoms,
        moldescriptor_file,
        orientation_indices_dict,
        output_file=output_file,
        cutoff_distance=cutoff_distance,
        mode=output_mode
    )

    if output_file is not None:
        save_neighbor_list(neighbor_list, output_file)
        print(f"Neighbor list saved to '{output_file}.npz'.")
        print(f"Total time taken: {time() - start_time:.2f} seconds.")
