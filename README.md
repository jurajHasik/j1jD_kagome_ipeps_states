# Infinite PEPS and PESS states for Heisenberg model with Dzyaloshinskii–Moriya interaction on a Kagome lattice

This repository contains a dataset of optimized 2D tensor network states  representing ground states of antiferromagnetic Heisenberg model with Dzyaloshinskii–Moriya interaction (DMI) on Kagome lattice, obtained in [SciPost Phys. 14, 139 (2023)](https://scipost.org/SciPostPhys.14.6.139).

This dataset contains two families of states and their observables:

* **IPEPS** - a single rank-5 tensor accounts for 3 spins on each up-pointing triangle of Kagome lattice
* **IPESS** - a set of five rank-3 tensors together accounting for 3 spins on each up-pointing triangle: two tensors without physical degrees of freedom and three *bond* tensor, one for each spin on the up-pointing triangle

For each dataset the states are stored in the corresponding folder, i.e.  ``IPEPS/J<nearest-neighbour-exchange>_JD<DMI-strength>`` or ``IPESS/J<nearest-neighbour-exchange>_JD<DMI-strength>``. States, or more specifically the tensors, are stored in plain text format (JSON). Below, we describe the individual datasets in more detail. 

## iPEPS

The single-site iPEPS ansatz adopted in [SciPost Phys. 14, 139 (2023)](https://scipost.org/SciPostPhys.14.6.139) coarse-grains Kagome lattice to an effective square lattice by 
grouping three spin-1/2's on each up-pointing triangle into a single degree of freedom with local Hilbert space ℋ = ⊗<sub>3</sub> ℋ(spin-1/2) of dimension 8.

The on-site tensor *a<sup>s</sup><sub>uldr</sub>* adopts the following index convention

       u
       |
    l--\                    
        \
        s0--s2--r      = a[s,u,l,d,r]
         | /
         |/   <- up-pointing triangle, where s0, s1, and s2 denote position of spins
        s1
         |               
         d               

where physical index *s* enumerates states of three spin-1/2's in the basis s<sub>0</sub>⊗s<sub>1</sub>⊗s<sub>2</sub>. The indices *u,l,d,r* of the virtual degrees of freedom, each of bond dimension D, are associated with up, left, down, and right directions on a square lattice.

Finally, the Kagome lattice is made up from the pattern above. Replacing each up-pointing triangle by tensor *a* and contracting the neighbouring tensors over their virtual indices 
creates an iPEPS tensor network defined on a square lattice
                         
             |            
           --\          
              \           |
              s0--s2--  --\    
               | /         \            |  |
               |/          s0--s2-- = --a--a--
              s1            | /         |  |
               |            |/        --a--a--
               |           s1           |  |
             --\            |
                \           |
                s0--s2-- --\      
                 | /        \
                 |/         s0--s2--
                s1          | /
                 |          |/
                           s1
                            |

Note that not all edges of Kagome lattice are explicitly shown here.
Due to coarse-graining, some nearest-neighbour bonds (edges) of Kagome lattice become next-nearest
neighbour bonds on the effective square lattice.

## iPESS

iPESS ansatz endows the rank-5 on-site tensor *a<sup>s</sup><sub>uldr</sub>* with more structure,
further constraining the variational freedom. 

The on-site tensor is obtained by contraction of 5 smaller rank-3 tensors:

* two rank-3 *trivalent* tensors *T<sub>u</sub>* and *T<sub>d</sub>* residing within up- and down-pointing triangles. These tensors have only virtual indices of bond dimension D.
* three rank-3 *bond* tensors *B<sub>a</sub>*, *B<sub>b</sub>*, and *B<sub>c</sub>* residing on corners shared between different triangles of Kagome lattice. Each bond tensor is associated to one of the three spins of an up-pointing triangle and carries a physical index corresponding to its spin-1/2 degree of freedom.

To obtain tensor *a<sup>s</sup><sub>uldr</sub>*, contract the iPESS tensors as follows

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
                       /  \                   where index conventions for iPESS tensors are:
                    1(k)  2(x)              
                  1(k)     1(x)               T_u[i,u,l]
                   /          \               T_d[j,k,x]
           0(n)==B_b          B_a==0(o)       B_a[o,x,r]
                 /              \             B_b[n,k,d]
              2(d)             2(r)           B_c[m,j,i]

    # Simple snippet demonstrating the contraction and reshaping of the result into an on-site tensor a^s_uldr

    A= einsum('iul,mji,jkx,nkd,oxr->mnouldr', T_u, B_c, T_d, B_b, B_a)
    A= A.reshape(8,D,D,D,D) # where D is the bond dimension

here, the first index of each bond tensor is physical index. Neighbouring trivalent tensors are connected by contractions with bond tensors creating the iPESS tensor network.

> This choice represents the geometry of Kagome lattice more faithfully, treating the up- and down-pointing triangles on equal footing. Moreover, it allows for imposing further point-group symmetries on trivalent and/or bond tensors. It is also common starting point for imposing internal symmetries on iPESS tensor

## Observables

Each state is accompanied with a simple plain-text ``*.dat`` file containing selected observables. These are evaluated using corner transfer matrix algorithm implemented in [``peps-torch``](https://github.com/jurajHasik/peps-torch). For a set of increasing environment bond dimensions χ, they are

* energy per site of Heisenberg antiferromagnet with DMI (see Eq.1 of  [SciPost Phys. 14, 139 (2023)](https://scipost.org/SciPostPhys.14.6.139))
* energies of up- and down-pointing triangles. Their difference indicates inversion symmetry breaking
* magnetization m=|⟨S⟩|, with S=(S<sup>z</sup>,S<sup>x</sup>,S<sup>y</sup>) the vector of spin-1/2 operators, for each of the three spins within up-pointing triangle
* individual spin components S<sup>z</sup>, S<sup>+</sup>, S<sup>-</sup> for each spin
* Nearest-neighbour spin-spin correlations of spins within down- ⟨S.S⟩<sub>down,01</sub>, ⟨S.S⟩<sub>down,02</sub>, ⟨S.S⟩<sub>down,12</sub> and up-pointing triangle ⟨S.S⟩<sub>up,01</sub>, ⟨S.S⟩<sub>up,02</sub>, ⟨S.S⟩<sub>up,12</sub>
* leading eigenvalues λ<sub>0,x</sub>, λ<sub>1,x</sub>, λ<sub>2,x</sub> of (width-1) horizontal and λ<sub>0,y</sub>, λ<sub>1,y</sub>, λ<sub>2,y</sub> of vertical transfer matrix. The spectra are normalized (λ<sub>0,*</sub>=1) and the leading correlation length can be obtained as ξ<sub>x</sub> = -1/ln(λ<sub>1,x</sub>) and ξ<sub>y</sub> = -1/ln(λ<sub>1,y</sub>)

## Reading and exporting states

To parse the states use the Python script ``ipeps_io.py`` which can export the dense tensor *a<sup>s</sup><sub>uldr</sub>* to either NumPy's *.npz format or MATLAB's *.mat format (requires SciPy)

    python ipeps_io.py --instate path/to/json/file --format mat [--out optional/name/for/exported/file]
    python ipeps_io.py --instate path/to/json/file --format npz [--out optional/name/for/exported/file]

Or access the (NumPy) tensor directly in the interactive mode

    >>> from ipeps_io import load_from_pepstorch_json_dense
    >>> A=load_from_pepstorch_json_dense("IPEPS/J1.0_JD0.0/IPEPS_J1.0_JD0.0_D3_chi_opt40.json")
    >>> type(A)
    <class 'numpy.ndarray'>
    >>> A.shape
    (8, 3, 3, 3, 3)
    >>> A[:,0,0,0,0]
    array([-0.00782555,  0.12528113, -0.13526041, -0.01993145, -0.02498749,
        0.11935324, -0.12645546, -0.00031023])

When exporting iPESS states, both usual rank-5 on-site tensor  *a<sup>s</sup><sub>uldr</sub>* is exported
as well as the individual iPESS tensors, stored under the keys defined in section above.

    >>> from ipeps_io import load_peps_from_json_dense, load_pess_from_json_dense
    >>> A,ipess_tensors=load_pess_from_json_dense("IPESS/J1.0_JD0.0/IPESS_J1.0_JD0.0_D4_chi_opt20.json")
    >>> ipess_tensors.keys()
    dict_keys(['T_u', 'T_d', 'B_c', 'B_a', 'B_b'])
    >>> ipess_tensors['T_u'][:,:,0]
    array([[-0.47092122,  1.39894781,  0.39612967, -0.89794985],
        [ 0.45824881, -0.77460775, -0.83686949, -0.71734594],
        [ 0.24671555, -0.47313889, -3.47609289,  0.39378996],
        [ 1.03717042,  0.58974852,  0.09221363, -0.84499218]])