from jtnn.mol_tree import MolTree

vocab = set()

with open("train.txt") as f:
    for i, line in enumerate(f):
        smi = line.strip().split()[0]
        if not smi:
            continue

        try:
            mol = MolTree(smi)
            for node in mol.nodes:
                vocab.add(node.smiles)
        except Exception as e:
            print(f"Failed at line {i}: {smi}")
            continue

        # 👇 progress print
        if i % 1000 == 0:
            print(f"Processed {i} molecules | vocab size: {len(vocab)}")

with open("vocab.txt", "w") as f:
    for v in sorted(vocab):
        f.write(v + "\n")

print("Final vocab size:", len(vocab))