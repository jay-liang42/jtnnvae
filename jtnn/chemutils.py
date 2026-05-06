import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

MST_MAX_WEIGHT = 100
MAX_NCAND = 2000


# -------------------------
# Basic utilities
# -------------------------

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        return get_mol(smiles)
    except Exception:
        return None


# -------------------------
# Stereo decoding
# -------------------------

def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    if mol is None:
        return []

    dec_isomers = list(EnumerateStereoisomers(mol))

    smiles3D = []
    for m in dec_isomers:
        try:
            smiles3D.append(Chem.MolToSmiles(m, isomericSmiles=True))
        except:
            continue

    # FIX: safeguard empty case
    if len(dec_isomers) == 0:
        return smiles3D

    chiralN = [
        atom.GetIdx()
        for atom in dec_isomers[0].GetAtoms()
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
        and atom.GetSymbol() == "N"
    ]

    if chiralN:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(
                    Chem.rdchem.ChiralType.CHI_UNSPECIFIED
                )
            try:
                smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))
            except:
                pass

    return list(set(smiles3D))


# -------------------------
# Molecule copy/edit helpers
# -------------------------

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol()

    for atom in mol.GetAtoms():
        new_mol.AddAtom(copy_atom(atom))

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        new_mol.AddBond(a1, a2, bond.GetBondType())

    return new_mol


def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    return sanitize(new_mol)


# -------------------------
# Tree decomposition (FIXED iteritems + stability)
# -------------------------

def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for _ in range(n_atoms)]
    for i, c in enumerate(cliques):
        for atom in c:
            nei_list[atom].append(i)

    # merge overlapping rings
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2:
                    continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i] = list(set(cliques[i] + cliques[j]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]

    nei_list = [[] for _ in range(n_atoms)]
    for i, c in enumerate(cliques):
        for atom in c:
            nei_list[atom].append(i)

    edges = defaultdict(int)

    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue

        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]

        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1

        elif len(rings) > 2:
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1

        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    edges[(c1, c2)] = max(edges[(c1, c2)], len(inter))

    # FIX: Python3 iteritems → items
    edges = [(u, v, MST_MAX_WEIGHT - w) for (u, v), w in edges.items()]

    if len(edges) == 0:
        return cliques, []

    row, col, data = zip(*edges)
    n_clique = len(cliques)

    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)

    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    return cliques, edges


# -------------------------
# Comparisons
# -------------------------

def atom_equal(a1, a2):
    return (
        a1.GetSymbol() == a2.GetSymbol()
        and a1.GetFormalCharge() == a2.GetFormalCharge()
    )


def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    b2 = (b2.GetEndAtom(), b2.GetBeginAtom()) if reverse else (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])


# -------------------------
# Attachment helpers (kept same logic, safer indexing)
# -------------------------

def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):
    prev_nids = [node.nid for node in prev_nodes]

    for nei_node in prev_nodes + neighbors:
        nei_id, nei_mol = nei_node.nid, nei_node.mol
        amap = nei_amap[nei_id]

        for atom in nei_mol.GetAtoms():
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        if nei_mol.GetNumBonds() == 0:
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())

        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtomIdx()]
                a2 = amap[bond.GetEndAtomIdx()]

                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
                elif nei_id in prev_nids:
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())

    return ctr_mol


def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei.nid: {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom

    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()


# -------------------------
# enum_assemble fix (critical stability + RDKit safety)
# -------------------------

def enum_assemble(node, neighbors, prev_nodes=None, prev_amap=None):
    prev_nodes = prev_nodes or []
    prev_amap = prev_amap or []

    all_attach_confs = []
    singletons = [
        nei.nid for nei in neighbors + prev_nodes
        if nei.mol.GetNumAtoms() == 1
    ]

    def search(cur_amap, depth):
        if len(all_attach_confs) > MAX_NCAND:
            return
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        nei_node = neighbors[depth]
        cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)

        cand_smiles = set()
        candidates = []

        for amap in cand_amap:
            cand_mol = local_attach(node.mol, neighbors[:depth + 1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue

            smiles = get_smiles(cand_mol)
            if smiles in cand_smiles:
                continue

            cand_smiles.add(smiles)
            candidates.append(amap)

        if not candidates:
            return

        for new_amap in candidates:
            search(new_amap, depth + 1)

    search(prev_amap, 0)

    cand_smiles = set()
    candidates = []

    for amap in all_attach_confs:
        cand_mol = local_attach(node.mol, neighbors, prev_nodes, amap)
        if cand_mol is None:
            continue

        mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
        if mol is None:
            continue

        smiles = Chem.MolToSmiles(mol)
        if smiles in cand_smiles:
            continue

        cand_smiles.add(smiles)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        candidates.append((smiles, mol, amap))

    return candidates