import json
import logging
import numpy as np
import argparse
from typing import Union, Dict
logger = logging.getLogger("ipeps_io")


parser= argparse.ArgumentParser(description='',allow_abbrev=False)
# additional model-dependent arguments
parser.add_argument("--instate", type=str, default=None, help="state to parse")
parser.add_argument("--out",type=str, default=None, help="output file name")
parser.add_argument("--format", type=str, default="npz", help="desired format",\
    choices=["npz","mat"])
args, unknown_args= parser.parse_known_args()

def read_bare_json_tensor_np_legacy(json_obj):
    t= json_obj

    # 0) find the dimensions of indices
    if "dims" in t.keys():
        dims= t["dims"]
    else:
        # assume all auxiliary indices have the same dimension and tensor is rank-5
        dims= [t["physDim"], t["auxDim"], t["auxDim"], \
            t["auxDim"], t["auxDim"]]
        
    # 1) find dtype, else assuming rank-5 tensor check length of an entries row (legacy)
    dtype_str=None
    if "dtype" in t.keys():
        dtype_str= json_obj["dtype"].lower()
    elif len(dims)==5:
        if len(t["entries"][0].split()) == 7:
            dtype_str= "complex128"
        elif len(t["entries"][0].split()) == 6:
            dtype_str= "float64"
    assert dtype_str in ["float64","complex128"], "Invalid dtype "+dtype_str

    # 2) fill the tensor with elements from the list "entries"
    # which list the non-zero tensor elements in the following
    # notation. Dimensions are indexed starting from 0
    # 
    # index (integer) of physDim, left, up, right, down, (float) Re, Im
    #                             (or generic auxilliary inds ...)
    X= np.zeros(dims, dtype=dtype_str)
    if dtype_str=="complex128":
        for entry in t["entries"]:
            l = entry.split()
            X[tuple(int(i) for i in l[:-2])]=float(l[-2])+float(l[-1])*1.0j
    else:
        for entry in t["entries"]:
            l= entry.split()
            k= 1 if len(l)==len(dims)+1 else 2
            X[tuple(int(i) for i in l[:-k])]+=float(l[-k])
    
    return X

def load_peps_from_json_dense(state)->np.ndarray:
    r"""
    """
    with open(state) as f:
        json_s= json.load(f)

    logger.info("Processed format detected: peps-torch single dense tensor")
    assert len(json_s["sites"]) == 1, "Multisite dense not yet implemented"
    site = json_s["sites"][0]
    A= read_bare_json_tensor_np_legacy(site)

    return A

def load_pess_from_json_dense(state)->Union[np.ndarray,Dict[str,np.ndarray]]:
    r"""
    """
    with open(state) as f:
        json_s= json.load(f)

    assert set(('T_u','T_d','B_a','B_b','B_c'))==set(list(json_s["ipess_tensors"].keys())),\
        "missing ipess tensors"
    ipess_tensors= {key: read_bare_json_tensor_np_legacy(t) for key,t in json_s["ipess_tensors"].items()}
    A= build_onsite_tensors(ipess_tensors)
    return A,ipess_tensors

def build_onsite_tensors(ipess_tensors)->np.ndarray:
        r"""
        :return: on-site tensor of underlying IPEPS
        :rtype: np.ndarray

        Build rank-5 on-site tensor by contracting the iPESS tensors.

                    2(l)   1(u)
                       \   /
                        T_u                      u
                         |                       |
                         0(i)                 l--\
                         2(i)                     \
                         |                         m--o--r
                         B_c==0(m)       =         | /        = a[m,n,o,u,l,d,r] = a[s=(m,n,o),u,l,d,r]
                         |                         |/
                         1(j)                      n
                         0(j)                      |
                         |                         d
                        T_d
                       /  \
                    1(k)  2(x)
                  1(k)     1(x)
                   /          \
           0(n)==B_b          B_a==0(o)
                 /              \
              2(d)             2(r)
        """
        A= np.einsum('iul,mji,jkx,nkd,oxr->mnouldr', ipess_tensors['T_u'],
            ipess_tensors['B_c'], ipess_tensors['T_d'], ipess_tensors['B_b'], \
            ipess_tensors['B_a'])
        total_phys_dim= ipess_tensors['B_a'].shape[0]*ipess_tensors['B_b'].shape[0]\
            *ipess_tensors['B_c'].shape[0]
        A= A.reshape([total_phys_dim]+[ipess_tensors['T_u'].shape[1], \
            ipess_tensors['T_u'].shape[2], ipess_tensors['B_b'].shape[2], \
            ipess_tensors['B_a'].shape[2]])
        return A

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    state_type=None
    with open(args.instate) as f:
        in_json= json.load(f)
        if "sites" in in_json:
            state_type="IPEPS"
        elif "ipess_tensors" in in_json:
            state_type="IPESS"
        else:
            raise Exception("No \"sites\" nor \"ipess_tensors\" field in instate")
    if args.format=="npz":
        outf= args.out if not (args.out is None) else "A.npz"
        if state_type=="IPEPS":
            np.savez(outf, A=load_peps_from_json_dense(args.instate))
        elif state_type=="IPESS":
            A,ipess_tensors= load_pess_from_json_dense(args.instate)
            np.savez(outf,A=A, **ipess_tensors)
    elif args.format=="mat":
        from scipy.io import savemat
        outf= args.out if not (args.out is None) else "A.mat"
        if state_type=="IPEPS":
            savemat(outf, {"A": load_peps_from_json_dense(args.instate)})
        elif state_type=="IPESS":
            A,ipess_tensors= load_pess_from_json_dense(args.instate)
            savemat(outf, {'A': A, **ipess_tensors})
        