from pathlib import Path
import numpy as np


def reorder_xyz_by_residue_type(
    xyz_file,
    moldescriptor,
    residue_types_order,
    output_file=None,
):
    """
    Reorder an XYZ file according to residue types.

    Parameters
    ----------
    xyz_file : str or Path
        Input XYZ file.
    moldescriptor : list[dict]
        Output of read_moldescriptor().
        Each dict must contain 'name', 'atom_count', and 'elements'.
    residue_types_order : list[int]
        Desired order of residue types.
        Example: [3, 4, 1, 2].
    output_file : str or Path, optional
        Output XYZ file. If None, the input file is overwritten.

    Returns
    -------
    None
    """

    xyz_file = Path(xyz_file)

    if output_file is None:
        output_file = xyz_file
    else:
        output_file = Path(output_file)

    # --------------------------------------------------
    # Read XYZ
    # --------------------------------------------------

    with open(xyz_file, "r") as f:
        lines = f.readlines()

    number_of_atoms = int(lines[0].strip().split()[0])
    header = lines[0]

    atom_lines = lines[2:]

    if len(atom_lines) != number_of_atoms:
        raise ValueError(
            f"XYZ header specifies {number_of_atoms} atoms, "
            f"but {len(atom_lines)} atom lines were found."
        )

    # --------------------------------------------------
    # Create mapping from residue type to descriptor
    # --------------------------------------------------

    descriptor_by_type = {
        residue_type: moldescriptor[residue_type - 1]
        for residue_type in residue_types_order
    }

    # --------------------------------------------------
    # Split XYZ into individual residues
    # --------------------------------------------------

    residues = []

    atom_index = 0
    residue_index = 0

    while atom_index < number_of_atoms:

        matching_residue = None

        # Identify the next residue by matching its element sequence
        for residue_type in residue_types_order:

            descriptor = descriptor_by_type[residue_type]
            atom_count = descriptor["atom_count"]

            candidate_lines = atom_lines[
                atom_index:atom_index + atom_count
            ]

            candidate_elements = [
                line.split()[0]
                for line in candidate_lines
            ]

            if candidate_elements == descriptor["elements"]:

                matching_residue = residue_type
                break

        if matching_residue is None:
            raise ValueError(
                f"Could not identify residue at atom index "
                f"{atom_index}. Check the XYZ atom order "
                f"and moldescriptor."
            )

        atom_count = descriptor_by_type[
            matching_residue
        ]["atom_count"]

        residue_lines = atom_lines[
            atom_index:atom_index + atom_count
        ]

        residues.append(
            {
                "type": matching_residue,
                "index": residue_index,
                "lines": residue_lines,
            }
        )

        atom_index += atom_count
        residue_index += 1

    # --------------------------------------------------
    # Reorder residues
    # --------------------------------------------------

    reordered_residues = []

    for residue_type in residue_types_order:

        residues_of_type = [
            residue
            for residue in residues
            if residue["type"] == residue_type
        ]

        reordered_residues.extend(residues_of_type)

    # --------------------------------------------------
    # Write reordered XYZ
    # --------------------------------------------------

    reordered_atom_lines = []

    for residue in reordered_residues:
        reordered_atom_lines.extend(residue["lines"])

    with open(output_file, "w") as f:

        f.write(header + "\n")

        f.writelines(reordered_atom_lines)

    print(
        f"Reordered {len(residues)} residues "
        f"and {number_of_atoms} atoms."
    )