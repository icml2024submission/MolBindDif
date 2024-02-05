import pandas as pd
import numpy as np
import Bio
import Bio.PDB
import Bio.SeqRecord
import subprocess



def rigidFrom3points(x1, x2, x3):
    v1 = x3-x2
    v2 = x1-x2
    e1 = v1/np.linalg.norm(v1)
    u2 = v2-e1*(e1.T@v2)
    e2 = u2/np.linalg.norm(u2)
    e3 = np.cross(e1,e2)
    return(x2,e1,e2,e3)


def createAchain(chain, idxs):

    cols = ['id', 'residue', 'binded', 'base_t1', 'base_t2', 'base_t3'] + [f'base_e1{i}' for i in range(1,4)]+ [f'base_e2{i}' for i in range(1,4)]+ [f'base_e2{i}' for i in range(1,4)]
    coordA = pd.DataFrame(columns=cols)

    for r in chain:
            n,ca,c = [None],[None],[None]
            for atom in r:
                if atom.id == "N": n=np.array([x for x in atom.get_vector()])
                if atom.id == "CA": ca=np.array([x for x in atom.get_vector()])
                if atom.id == "C": c=np.array([x for x in atom.get_vector()])

            if n[0]==None or ca[0]==None or c[0]==None:
                continue
            else:
                rb = rigidFrom3points(n, ca, c)
                coordA.loc[len(coordA)] = np.hstack((np.array([r.id[1], r.resname, int(r.id[1] in idxs)]), rb[0],rb[1],rb[2],rb[3]))
            
    return coordA


def createBchain(chain, idxs, expanded_idxs):

    cols = ['id', 'residue', 'binded', 'base_t1', 'base_t2', 'base_t3'] + [f'base_e1{i}' for i in range(1,4)]+ [f'base_e2{i}' for i in range(1,4)]+ [f'base_e2{i}' for i in range(1,4)]
    cols = cols + ['rib_t1', 'rib_t2', 'rib_t3'] + [f'rib_e1{i}' for i in range(1,4)]+ [f'rib_e2{i}' for i in range(1,4)]+ [f'rib_e2{i}' for i in range(1,4)]
    coordB = pd.DataFrame(columns=cols)

    for r in chain:
        if r.id[1] in expanded_idxs:
            if r.resname == 'G' or r.resname == 'A':
                n7,n9,n3,c4,c3,c2 = [None],[None],[None],[None],[None],[None]
                for atom in r:
                    if atom.id == "N7": n7=np.array([x for x in atom.get_vector()])
                    if atom.id == "N9": n9=np.array([x for x in atom.get_vector()])
                    if atom.id == "N3": n3=np.array([x for x in atom.get_vector()])
                    if atom.id == "C4'": c4=np.array([x for x in atom.get_vector()])
                    if atom.id == "C3'": c3=np.array([x for x in atom.get_vector()])
                    if atom.id == "C2'": c2=np.array([x for x in atom.get_vector()])


                if n7[0]==None or n9[0]==None or n3[0]==None or c4[0]==None or c3[0]==None or c2[0]==None:
                    continue
                else:
                    rb1 = rigidFrom3points(n7, n9, n3)
                    rb2 = rigidFrom3points(c4, c3, c2)
                    coordB.loc[len(coordB)] = np.hstack((np.array([r.id[1], r.resname, int(r.id[1] in idxs)]), rb1[0],rb1[1],rb1[2],rb1[3], rb2[0],rb2[1],rb2[2],rb2[3]))

            if r.resname == 'U' or r.resname == 'C':
                o2,n1,n3,c4,c3,c2 = [None],[None],[None],[None],[None],[None]
                for atom in r:
                    if atom.id == "O2": o2=np.array([x for x in atom.get_vector()])
                    if atom.id == "N1": n1=np.array([x for x in atom.get_vector()])
                    if atom.id == "N3": n3=np.array([x for x in atom.get_vector()])
                    if atom.id == "C4'": c4=np.array([x for x in atom.get_vector()])
                    if atom.id == "C3'": c3=np.array([x for x in atom.get_vector()])
                    if atom.id == "C2'": c2=np.array([x for x in atom.get_vector()])
                
                if o2[0]==None or n1[0]==None or n3[0]==None or c4[0]==None or c3[0]==None or c2[0]==None:
                    continue
                else:
                    rb1 = rigidFrom3points(o2, n1, n3)
                    rb2 = rigidFrom3points(c4, c3, c2)
                    coordB.loc[len(coordB)] = np.hstack((np.array([r.id[1], r.resname, int(r.id[1] in idxs)]), rb1[0],rb1[1],rb1[2],rb1[3], rb2[0],rb2[1],rb2[2],rb2[3]))

            
    return coordB


def select_idxs_B(idxs, rr):
    expanded_idxs = []
    for idx in idxs:
        selected = rr[(rr['rnum1'] == idx) | (rr['rnum2'] == idx)]
        neib1 = list(selected['rnum1'].values) +  list(selected['rnum2'].values)
        neib1 = list(filter(lambda a: a != idx, neib1))
        neib2 = []
        for n in neib1:
            selected = rr[(rr['rnum1'] == n) | (rr['rnum2'] == n)]
            neib2 += list(selected['rnum1'].values) +  list(selected['rnum2'].values)
        expanded_idxs += neib2
    expanded_idxs = list(set(expanded_idxs))
    expanded_idxs.sort()
    return expanded_idxs



def create_data(dir, name):
    link, pdbid = name, name.split('-')[1]

    subprocess.run(['./create_contacts.sh', link, dir], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    rp_contacts = pd.read_csv(f'{dir}/data/rp_{link}.tsv', sep='\t')
    rr_contacts = pd.read_csv(f'{dir}/data/rr_{link}.tsv', sep='\t')

    idxsA = rp_contacts['ID1_resSeq'].values

    idxsB = rp_contacts['ID2_resSeq'].values
    expanded_idxsB = select_idxs_B(idxsB, rr_contacts)
    
    if len(expanded_idxsB) < 200:
        pdbparser = Bio.PDB.PDBParser(QUIET=True)
        struct = pdbparser.get_structure(pdbid, f'{dir}/data/{link}.pdb')
        
        try:
            coordA = createAchain(struct[0]["A"], idxsA)
            coordB = createBchain(struct[0]["B"], idxsB, expanded_idxsB)

            if len(coordB)>0 and len(coordA)>0:
                coordA.to_csv(f'{dir}/data/A_{link}.csv', index=False)
                coordB.to_csv(f'{dir}/data/B_{link}.csv', index=False)
            else:
                print(f'No contacts have been found in {link}!')
        except: 
            pass
    
    return f'{dir}/data/A_{link}.csv', f'{dir}/data/B_{link}.csv'

