# these are some data dictionaries that are used throughout the addon
# Usually it's for converting things like Amino Acid names into their
# numerical representation in alphabetical order, or converting the name
# of an element to its atomic number etc


# vdw_radii given in picometres
elements = {
        "H"  : {"atomic_number" :  1, 'vdw_radii': 120, "name" : "Hydrogen"    },
        "He" : {"atomic_number" :  2, 'vdw_radii': 140, "name" : "Helium"      },
        "Li" : {"atomic_number" :  3, 'vdw_radii': 182, "name" : "Lithium"     },
        "Be" : {"atomic_number" :  4, 'vdw_radii': 153, "name" : "Beryllium"   },
        "B"  : {"atomic_number" :  5, 'vdw_radii': 192, "name" : "Boron"       },
        "C"  : {"atomic_number" :  6, 'vdw_radii': 170, "name" : "Carbon"      },
        "N"  : {"atomic_number" :  7, 'vdw_radii': 155, "name" : "Nitrogen"    },
        "O"  : {"atomic_number" :  8, 'vdw_radii': 152, "name" : "Oxygen"      },
        "F"  : {"atomic_number" :  9, 'vdw_radii': 147, "name" : "Fluorine"    },
        "Ne" : {"atomic_number" : 10, 'vdw_radii': 154, "name" : "Neon"        },
        "Na" : {"atomic_number" : 11, 'vdw_radii': 227, "name" : "Sodium"      },
        "Mg" : {"atomic_number" : 12, 'vdw_radii': 173, "name" : "Magnesium"   },
        "Al" : {"atomic_number" : 13, 'vdw_radii': 184, "name" : "Aluminium"   },
        "Si" : {"atomic_number" : 14, 'vdw_radii': 210, "name" : "Silicon"     },
        "P"  : {"atomic_number" : 15, 'vdw_radii': 180, "name" : "Phosphorus"  },
        "S"  : {"atomic_number" : 16, 'vdw_radii': 180, "name" : "Sulfur"      },
        "Cl" : {"atomic_number" : 17, 'vdw_radii': 175, "name" : "Chlorine"    },
        "Ar" : {"atomic_number" : 18, 'vdw_radii': 188, "name" : "Argon"       },
        "K"  : {"atomic_number" : 19, 'vdw_radii': 275, "name" : "Potassium"   },
        "Ca" : {"atomic_number" : 20, 'vdw_radii': 231, "name" : "Calcium"     },
        "Sc" : {"atomic_number" : 21, 'vdw_radii': 211, "name" : "Scandium"    },
        "Ti" : {"atomic_number" : 22, 'vdw_radii': 147, "name" : "Titanium"    }, # 'Metallic' radii as vdw isn't listed, unsure what this means (not a chemist)
        "V"  : {"atomic_number" : 23, 'vdw_radii': 134, "name" : "Vanadium"    }, # 'Metallic' radii as vdw isn't listed, unsure what this means (not a chemist)
        "Cr" : {"atomic_number" : 24, 'vdw_radii': 128, "name" : "Chromium"    }, # 'Metallic' radii as vdw isn't listed, unsure what this means (not a chemist)
        "Mn" : {"atomic_number" : 25, 'vdw_radii': 127, "name" : "Manganese"   }, # 'Metallic' radii as vdw isn't listed, unsure what this means (not a chemist)
        "Fe" : {"atomic_number" : 26, 'vdw_radii': 126, "name" : "Iron"        }, # 'Metallic' radii as vdw isn't listed, unsure what this means (not a chemist)
        "Co" : {"atomic_number" : 27, 'vdw_radii': 125, "name" : "Cobalt"      }, # 'Metallic' radii as vdw isn't listed, unsure what this means (not a chemist)
        "Ni" : {"atomic_number" : 28, 'vdw_radii': 163, "name" : "Nickel"      },
        "Cu" : {"atomic_number" : 29, 'vdw_radii': 140, "name" : "Copper"      },
        "Zn" : {"atomic_number" : 30, 'vdw_radii': 139, "name" : "Zinc"        },
        "Ga" : {"atomic_number" : 31, 'vdw_radii': 187, "name" : "Gallium"     },
        "Ge" : {"atomic_number" : 32, 'vdw_radii': 211, "name" : "Germanium"   },
        "As" : {"atomic_number" : 33, 'vdw_radii': 185, "name" : "Arsenic"     },
        "Se" : {"atomic_number" : 34, 'vdw_radii': 190, "name" : "Selenium"    },
        "Br" : {"atomic_number" : 35, 'vdw_radii': 185, "name" : "Bromine"     },
        "Kr" : {"atomic_number" : 36, 'vdw_radii': 202, "name" : "Krypton"     },
        "Rb" : {"atomic_number" : 37, 'vdw_radii': 303, "name" : "Rubidium"    },
        "Sr" : {"atomic_number" : 38, 'vdw_radii': 249, "name" : "Strontium"   },
        "Y"  : {"atomic_number" : 39, 'vdw_radii': 180, "name" : "Yttrium"     },
        "Zr" : {"atomic_number" : 40, 'vdw_radii': 160, "name" : "Zirconium"   },
        "Nb" : {"atomic_number" : 41, 'vdw_radii': 146, "name" : "Niobium"     },
        "Mo" : {"atomic_number" : 42, 'vdw_radii': 239, "name" : "Molybdenum"  },
        "Tc" : {"atomic_number" : 43, 'vdw_radii': 136, "name" : "Technetium"  },
        "Ru" : {"atomic_number" : 44, 'vdw_radii': 134, "name" : "Ruthenium"   },
        "Rh" : {"atomic_number" : 45, 'vdw_radii': 137, "name" : "Rhodium"     },
        "Pd" : {"atomic_number" : 46, 'vdw_radii': 144, "name" : "Palladium"   },
        "Ag" : {"atomic_number" : 47, 'vdw_radii': 100, "name" : "Silver"      },
        "Cd" : {"atomic_number" : 48, 'vdw_radii': 100, "name" : "Cadmium"     },
        "In" : {"atomic_number" : 49, 'vdw_radii': 100, "name" : "Indium"      },
        "Sn" : {"atomic_number" : 50, 'vdw_radii': 100, "name" : "Tin"         },
        "Sb" : {"atomic_number" : 51, 'vdw_radii': 100, "name" : "Antimony"    },
        "Te" : {"atomic_number" : 52, 'vdw_radii': 100, "name" : "Tellurium"   },
        "I"  : {"atomic_number" : 53, 'vdw_radii': 100, "name" : "Iodine"      },
        "Xe" : {"atomic_number" : 54, 'vdw_radii': 100, "name" : "Xenon"       },
        "Cs" : {"atomic_number" : 55, 'vdw_radii': 100, "name" : "Caesium"     },
        "Ba" : {"atomic_number" : 56, 'vdw_radii': 100, "name" : "Barium"      },
        "La" : {"atomic_number" : 57, 'vdw_radii': 100, "name" : "Lanthanum"   },
        "Ce" : {"atomic_number" : 58, 'vdw_radii': 100, "name" : "Cerium"      },
        "Pr" : {"atomic_number" : 59, 'vdw_radii': 100, "name" : "Praseodymium"},
        "Nd" : {"atomic_number" : 60, 'vdw_radii': 100, "name" : "Neodymium"   },
        "Pm" : {"atomic_number" : 61, 'vdw_radii': 100, "name" : "Promethium"  },
        "Sm" : {"atomic_number" : 62, 'vdw_radii': 100, "name" : "Samarium"    },
        "Eu" : {"atomic_number" : 63, 'vdw_radii': 100, "name" : "Europium"    },
        "Gd" : {"atomic_number" : 64, 'vdw_radii': 100, "name" : "Gadolinium"  },
        "Tb" : {"atomic_number" : 65, 'vdw_radii': 100, "name" : "Terbium"     },
        "Dy" : {"atomic_number" : 66, 'vdw_radii': 100, "name" : "Dysprosium"  },
        "Ho" : {"atomic_number" : 67, 'vdw_radii': 100, "name" : "Holmium"     },
        "Er" : {"atomic_number" : 68, 'vdw_radii': 100, "name" : "Erbium"      },
        "Tm" : {"atomic_number" : 69, 'vdw_radii': 100, "name" : "Thulium"     },
        "Yb" : {"atomic_number" : 70, 'vdw_radii': 100, "name" : "Ytterbium"   },
        "Lu" : {"atomic_number" : 71, 'vdw_radii': 100, "name" : "Lutetium"    },
        "Hf" : {"atomic_number" : 72, 'vdw_radii': 100, "name" : "Hafnium"     },
        "Ta" : {"atomic_number" : 73, 'vdw_radii': 100, "name" : "Tantalum"    },
        "W"  : {"atomic_number" : 74, 'vdw_radii': 100, "name" : "Tungsten"    },
        "Re" : {"atomic_number" : 75, 'vdw_radii': 100, "name" : "Rhenium"     },
        "Os" : {"atomic_number" : 76, 'vdw_radii': 100, "name" : "Osmium"      },
        "Ir" : {"atomic_number" : 77, 'vdw_radii': 100, "name" : "Iridium"     },
        "Pt" : {"atomic_number" : 78, 'vdw_radii': 100, "name" : "Platinum"    },
        "Au" : {"atomic_number" : 79, 'vdw_radii': 100, "name" : "Gold"        },
        "Hg" : {"atomic_number" : 80, 'vdw_radii': 100, "name" : "Mercury"     },
        "Tl" : {"atomic_number" : 81, 'vdw_radii': 100, "name" : "Thallium"    },
        "Pb" : {"atomic_number" : 82, 'vdw_radii': 100, "name" : "Lead"        },
        "Bi" : {"atomic_number" : 83, 'vdw_radii': 100, "name" : "Bismuth"     },
        "Po" : {"atomic_number" : 84, 'vdw_radii': 100, "name" : "Polonium"    },
        "At" : {"atomic_number" : 85, 'vdw_radii': 100, "name" : "Astatine"    },
        "Rn" : {"atomic_number" : 86, 'vdw_radii': 100, "name" : "Radon"       },
        "Fr" : {"atomic_number" : 87, 'vdw_radii': 100, "name" : "Francium"    },
        "Ra" : {"atomic_number" : 88, 'vdw_radii': 100, "name" : "Radium"      },
        "Ac" : {"atomic_number" : 89, 'vdw_radii': 100, "name" : "Actinium"    },
        "Th" : {"atomic_number" : 90, 'vdw_radii': 100, "name" : "Thorium"     },
        "Pa" : {"atomic_number" : 91, 'vdw_radii': 100, "name" : "Protactinium"},
        "U"  : {"atomic_number" : 92, 'vdw_radii': 100, "name" : "Uranium"     },
        "Np" : {"atomic_number" : 93, 'vdw_radii': 100, "name" : "Neptunium"   },
        "Pu" : {"atomic_number" : 94, 'vdw_radii': 100, "name" : "Plutonium"   },
        "Am" : {"atomic_number" : 95, 'vdw_radii': 100, "name" : "Americium"   },
        "Cm" : {"atomic_number" : 96, 'vdw_radii': 100, "name" : "Curium"      },
        "Bk" : {"atomic_number" : 97, 'vdw_radii': 100, "name" : "Berkelium"   },
        "Cf" : {"atomic_number" : 98, 'vdw_radii': 100, "name" : "Californium" },
        "Es" : {"atomic_number" : 99, 'vdw_radii': 100, "name" : "Einsteinium" },
        "Fm" : {"atomic_number" : 100, 'vdw_radii': 100, "name" : "Fermium"    },
        "Md" : {"atomic_number" : 101, 'vdw_radii': 100, "name" : "Mendelevium"},
        "No" : {"atomic_number" : 102, 'vdw_radii': 100, "name" : "Nobelium"   },
        "Lr" : {"atomic_number" : 103, 'vdw_radii': 100, "name" : "Lawrencium" }
    }

residues = {
    # unknown? Came up in one of the structures, haven't looked into it yet
    # TODO look into it!
    "UNK": {"res_name_num": -1, "res_type": "unkown",   "res_type_no": 1},
    
    # TODO implement an attribute that uses the residue types (basic / polar / acid etc)
    # 20 naturally occurring amino acids
    "ALA": {"res_name_num": 0,  "res_type": "apolar",   "res_type_no": 1},
    "ARG": {"res_name_num": 1,  "res_type": "basic",    "res_type_no": 1},
    "ASN": {"res_name_num": 2,  "res_type": "polar",    "res_type_no": 1},
    "ASP": {"res_name_num": 3,  "res_type": "acid",     "res_type_no": 1},
    "CYS": {"res_name_num": 4,  "res_type": "polar",    "res_type_no": 1}, #can act as acid, but mostly polar?
    "GLU": {"res_name_num": 5,  "res_type": "acid",     "res_type_no": 1},
    "GLN": {"res_name_num": 6,  "res_type": "polar",    "res_type_no": 1},
    "GLY": {"res_name_num": 7,  "res_type": "apolar",   "res_type_no": 1},
    "HIS": {"res_name_num": 8,  "res_type": "polar",    "res_type_no": 1}, #ambiguous
    "ILE": {"res_name_num": 9,  "res_type": "apolar",   "res_type_no": 1},
    "LEU": {"res_name_num": 10, "res_type": "apolar",   "res_type_no": 1},
    "LYS": {"res_name_num": 11, "res_type": "basic",    "res_type_no": 1},
    "MET": {"res_name_num": 12, "res_type": "apolar",   "res_type_no": 1},
    "PHE": {"res_name_num": 13, "res_type": "aromatic", "res_type_no": 1},
    "PRO": {"res_name_num": 14, "res_type": "apolar",   "res_type_no": 1},
    "SER": {"res_name_num": 15, "res_type": "polar",    "res_type_no": 1},
    "THR": {"res_name_num": 16, "res_type": "polar",    "res_type_no": 1},
    "TRP": {"res_name_num": 17, "res_type": "aromatic", "res_type_no": 1},
    "TYR": {"res_name_num": 18, "res_type": "aromatic", "res_type_no": 1},
    "VAL": {"res_name_num": 19, "res_type": "apolar",   "res_type_no": 1},
    
    # modified amino acids, unsure how to deal with but currently will label as the 
    # same res_name number
    "SNC": {"res_name_num": 15, "res_type": "unkown", "res_type_no": 1}, # S-nitrosylation of cysteine
    "MSE": {"res_name_num": 12, "res_type": "unkown", "res_type_no": 1}, # selenomethionine
    
    # add conventional AMBER FF residue names with different protonations
    "ASH": {"res_name_num": 3,  "res_type": "acid",     "res_type_no": 1},
    "CYM": {"res_name_num": 4,  "res_type": "polar",    "res_type_no": 1},
    "CYX": {"res_name_num": 4,  "res_type": "polar",    "res_type_no": 1},
    "GLH": {"res_name_num": 5,  "res_type": "acid",     "res_type_no": 1},
    "HID": {"res_name_num": 8,  "res_type": "polar",    "res_type_no": 1},
    "HIE": {"res_name_num": 8,  "res_type": "polar",    "res_type_no": 1},
    "HIP": {"res_name_num": 8,  "res_type": "polar",    "res_type_no": 1},
    "HYP": {"res_name_num": 8,  "res_type": "polar",    "res_type_no": 1},
    "LYN": {"res_name_num": 11, "res_type": "basic",    "res_type_no": 1},
    
    ## nucleic acids
    # DNA
    "DA": {"res_name_num": 30,  "res_type": "unkown",   "res_type_no": 1},
    "DC": {"res_name_num": 31,  "res_type": "unkown",   "res_type_no": 1},
    "DG": {"res_name_num": 32,  "res_type": "unkown",   "res_type_no": 1},
    "DT": {"res_name_num": 33,  "res_type": "unkown",   "res_type_no": 1},
    # RNA
    "A": {"res_name_num": 40,   "res_type": "unkown",   "res_type_no": 1},
    "C": {"res_name_num": 41,   "res_type": "unkown",   "res_type_no": 1},
    "G": {"res_name_num": 42,   "res_type": "unkown",   "res_type_no": 1},
    "U": {"res_name_num": 43,   "res_type": "unkown",   "res_type_no": 1}
    
    ## need to add some sugars and such here as well
}

lipophobicity = {
    'ALA': {'CB': 0.62,
            'C': -0.61,
            'CA': 0.02,
            'O': -0.58,
            'N': -0.49,
            'H': -0.5,
            'HA': -0.25,
            'HB1': 0.0,
            'HB2': 0.0,
            'HB3': 0.0,
            'OXT': 0.49},
    'ARG': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD': 0.45,
            'CG': 0.45,
            'CZ': -0.61,
            'N': -0.49,
            'NE': -0.49,
            'NH1': -0.14,
            'NH2': -0.69,
            'O': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HG2': 0.0,
            'HG3': 0.0,
            'HD2': -0.25,
            'HD3': -0.25,
            'HE': -0.5,
            'HH11': -0.5,
            'HH12': -0.5,
            'HH21': -0.5,
            'HH22': -0.5,
            '1HH1': -0.5,
            '2HH1': -0.5,
            '1HH2': -0.5,
            '2HH2': -0.5,
            'OXT': 0.49},
    'ASN': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.02,
            'CG': -0.61,
            'N': -0.49,
            'ND2': -0.14,
            'O': -0.58,
            'OD1': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HD21': -0.5,
            'HD22': -0.5,
            '1HD2': -0.5,
            '2HD2': -0.5,
            'OXT': 0.49},
    'ASP': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CG': -0.61,
            'N': -0.49,
            'O': -0.58,
            'OD1': -0.58,
            'OD2': 0.49,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'OXT': 0.49},
    'CYS': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'N': -0.49,
            'O': -0.58,
            'SG': 0.29,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'OXT': 0.49},
    'GLN': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD': -0.54,
            'CG': 0.45,
            'N': -0.49,
            'NE2': -0.14,
            'O': -0.58,
            'OE1': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HG2': 0.0,
            'HG3': 0.0,
            'HE21': -0.5,
            'HE22': -0.5,
            '1HE2': -0.5,
            '2HE2': -0.5,
            'OXT': 0.49},
    'GLU': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD': -0.54,
            'CG': 0.45,
            'N': -0.49,
            'O': -0.58,
            'OE1': -0.58,
            'OE2': 0.49,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HG2': 0.0,
            'HG3': 0.0,
            'OXT': 0.49},
    'GLY': {'C': -0.61,
            'CA': 0.45,
            'O': -0.58,
            'N': -0.57,
            'H': -0.5,
            'HA': -0.25,
            'HA2': 0.0,
            'HA3': 0.0,
            'OXT': 0.49},
    'HIS': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD2': 0.31,
            'CE1': 0.31,
            'CG': 0.1,
            'N': -0.49,
            'ND1': 0.08,
            'NE2': -1.14,
            'O': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HD1': -0.5,
            'HD2': -0.25,
            'HE1': -0.25,
            'OXT': 0.49},
    'ILE': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.02,
            'CD': 0.63,
            'CD1': 0.63,
            'CG1': 0.45,
            'CG2': 0.63,
            'N': -0.49,
            'O': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB': 0.0,
            'HG12': 0.0,
            'HG13': 0.0,
            'HG21': 0.0,
            'HG22': 0.0,
            'HG23': 0.0,
            'HD11': 0.0,
            'HD12': 0.0,
            'HD13': 0.0,
            '2HG1': 0.0,
            '3HG1': 0.0,
            '1HG2': 0.0,
            '2HG2': 0.0,
            '3HG2': 0.0,
            '1HD1': 0.0,
            '2HD1': 0.0,
            '3HD1': 0.0,
            'OXT': 0.49},
    'LEU': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD1': 0.63,
            'CD2': 0.63,
            'CG': 0.02,
            'N': -0.49,
            'O': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HG': 0.0,
            'HD11': 0.0,
            'HD12': 0.0,
            'HD13': 0.0,
            'HD21': 0.0,
            'HD22': 0.0,
            'HD23': 0.0,
            '1HD1': 0.0,
            '2HD1': 0.0,
            '3HD1': 0.0,
            '1HD2': 0.0,
            '2HD2': 0.0,
            '3HD2': 0.0,
            'OXT': 0.49},
    'LYS': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD': 0.45,
            'CE': 0.45,
            'CG': 0.45,
            'N': -0.49,
            'NZ': -1.07,
            'O': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HG2': 0.0,
            'HG3': 0.0,
            'HD2': 0.0,
            'HD3': 0.0,
            'HE2': -0.2,
            'HE3': -0.2,
            'HZ1': -0.5,
            'HZ2': -0.5,
            'HZ3': -0.5,
            'OXT': 0.49,
            },
    'MET': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CE': 0.63,
            'CG': 0.45,
            'N': -0.49,
            'O': -0.58,
            'SD': -0.30,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HG2': 0.0,
            'HG3': 0.0,
            'HE1': 0.0,
            'HE2': 0.0,
            'HE3': -0.5,
            'OXT': 0.49,
            },
    'PHE': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD1': 0.31,
            'CD2': 0.31,
            'CE1': 0.31,
            'CE2': 0.31,
            'CG': 0.1,
            'CZ': 0.31,
            'N': -0.49,
            'O': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HD1': 0.0,
            'HD2': 0.0,
            'HE1': 0.0,
            'HE2': 0.0,
            'HZ': 0.0,
            'OXT': 0.49},
    'PRO': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD': 0.45,
            'CG': 0.45,
            'N': -0.92,
            'O': -0.58,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HG2': 0.0,
            'HG3': 0.0,
            'HD2': -0.2,
            'HD3': -0.2,
            'OXT': 0.49, },
    'SER': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'N': -0.49,
            'O': -0.58,
            'OG': -0.99,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HG': 0.0,
            'OXT': 0.49, },
    'THR': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.02,
            'CG2': 0.62,
            'N': -0.49,
            'O': -0.58,
            'OG1': -0.9,
            'H': -0.5,
            'HA': -0.25,
            'HB': 0.0,
            'HG1': 0.0,
            'HG21': 0.0,
            'HG22': 0.0,
            'HG23': 0.0,
            '1HG2': 0.0,
            '2HG2': 0.0,
            '3HG2': 0.0,
            'OXT': 0.49, },
    'TRP': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD1': 0.31,
            'CD2': 0.25,
            'CE2': 0.25,
            'CE3': 0.31,
            'CG': 0.1,
            'CH2': 0.31,
            'CZ2': 0.31,
            'CZ3': 0.31,
            'N': -0.49,
            'NE1': 0.08,
            'O': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HE1': -0.5,
            'HD1': -0.2,
            'HE3': 0.0,
            'HZ2': 0.0,
            'HZ3': 0.0,
            'HH2': 0.0,
            'OXT': 0.49, },
    'TYR': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.45,
            'CD1': 0.31,
            'CD2': 0.31,
            'CE1': 0.31,
            'CE2': 0.31,
            'CG': 0.1,
            'CZ': 0.1,
            'N': -0.49,
            'O': -0.58,
            'OH': -0.17,
            'H': -0.5,
            'HA': -0.25,
            'HB2': 0.0,
            'HB3': 0.0,
            'HD1': 0.0,
            'HD2': 0.0,
            'HE1': 0.0,
            'HE2': 0.0,
            'HH': 0.0,
            'OXT': 0.49, },
    'VAL': {'C': -0.61,
            'CA': 0.02,
            'CB': 0.02,
            'CG1': 0.62,
            'CG2': 0.62,
            'N': -0.49,
            'O': -0.58,
            'H': -0.5,
            'HA': -0.25,
            'HB': 0.0,
            'HG11': 0.0,
            'HG12': 0.0,
            'HG13': 0.0,
            'HG21': 0.0,
            'HG22': 0.0,
            'HG23': 0.0,
            '1HG1': 0.0,
            '2HG1': 0.0,
            '3HG1': 0.0,
            '1HG2': 0.0,
            '2HG2': 0.0,
            '3HG2': 0.0,
            'OXT': 0.49, },
    'CA': {'CA': -1.0},
    'NAG': {'C1': 0.02,
            'C2': 0.02,
            'C3': 0.02,
            'C4': 0.02,
            'C5': 0.02,
            'C6': 0.31,
            'C7': -0.61,
            'C8': 0.62,
            'O1': -0.92,
            'O2': -0.92,
            'O3': -0.92,
            'O4': -0.92,
            'O5': -1.14,
            'O6': -0.99,
            'O7': -0.58,
            'N2': -0.49,
            'H2': -0.25,
            'HN2': -0.5, },
    'NDG': {'C1': 0.02,
            'C2': 0.02,
            'C3': 0.02,
            'C4': 0.02,
            'C5': 0.02,
            'C6': 0.031,
            'C7': -0.61,
            'C8': 0.62,
            'O1L': -0.9,
            'O3': -0.92,
            'O4': -0.92,
            'O': -1.14,
            'O6': -0.99,
            'O7': -0.58,
            'N2': -0.29, },
    'BMA': {'C1': 0.02,
            'C2': 0.02,
            'C3': 0.02,
            'C4': 0.02,
            'C5': 0.02,
            'C6': 0.31,
            'O1': -0.92,
            'O2': -0.92,
            'O3': -0.92,
            'O4': -0.92,
            'O5': -1.14,
            'O6': -0.58, },
    'MAN': {'C1': 0.02,
            'C2': 0.02,
            'C3': 0.02,
            'C4': 0.02,
            'C5': 0.02,
            'C6': 0.31,
            'O1': -0.92,
            'O2': -0.92,
            'O3': -0.92,
            'O4': -0.92,
            'O5': -1.14,
            'O6': -0.58, },
    'GAL': {'C1': 0.02,
            'C2': 0.02,
            'C3': 0.02,
            'C4': 0.02,
            'C5': 0.02,
            'C6': 0.31,
            'O1': -0.92,
            'O2': -0.92,
            'O3': -0.92,
            'O4': -0.92,
            'O5': -1.14,
            'O6': -0.58, },
    'NAN': {'C1': -0.61,
            'C2': 0.02,
            'C3': 0.62,
            'C4': 0.02,
            'C5': 0.02,
            'C6': 0.02,
            'C7': 0.02,
            'C8': 0.02,
            'C9': 0.31,
            'C10': -0.6,
            'C11': 0.62,
            'O1A': -0.2,
            'O1B': -0.4,
            'O2': -0.92,
            'O4': -0.92,
            'O6': -1.14,
            'O7': -0.92,
            'O8': -0.92,
            'O9': -0.7,
            'O10': -0.2,
            'N5': -0.49,
            'NH5': -0.5, },
    'DG': {'P': -0.94,
            'O1P': -0.7,
            'O2P': -0.22,
            "O5'": -0.5,
            "C5'": 0.45,
            "C4'": 0.02,
            "O4'": -1.14,
            "C1'": 0.02,
            "C2'": 0.45,
            "C3'": 0.02,
            "O3'": -0.92,
            'N9': -1.66,
            'C8': 0.31,
            'N7': -0.55,
            'C5': 0.25,
            'C6': 0.1,
            'O6': -0.58,
            'N1': -0.49,
            'C2': 0.1,
            'N2': -0.6,
            'N3': -0.07,
            'C4': -0.25, },
    'DA': {'P': -0.94,
            'O1P': -0.7,
            'O2P': -0.22,
            "O5'": -0.5,
            "C5'": 0.45,
            "C4'": 0.02,
            "O4'": -1.14,
            "C1'": 0.02,
            "C2'": 0.45,
            "C3'": 0.02,
            "O3'": -0.92,
            'N9': -1.66,
            'C8': 0.31,
            'N7': -0.55,
            'C5': 0.25,
            'C6': 0.1,
            'N6': -0.6,
            'N1': -0.49,
            'C2': 0.31,
            'N2': -0.6,
            'N3': -0.07,
            'C4': -0.25, },
    'DC': {'P': -0.94,
            'O1P': -0.7,
            'O2P': -0.22,
            "O5'": -0.5,
            "C5'": 0.45,
            "C4'": 0.02,
            "O4'": -1.14,
            "C1'": 0.02,
            "C2'": 0.45,
            "C3'": 0.02,
            "O3'": -0.92,
            'N1': -1.66,
            'C2': 0.1,
            'O2': -0.58,
            'N3': -0.29,
            'C4': 0.1,
            'N4': -0.6,
            'C5': 0.31,
            'C6': 0.31},
    'DT': {'P': -0.94,
            'O1P': -0.7,
            'O2P': -0.22,
            "O5'": -0.5,
            "C5'": 0.45,
            "C4'": 0.02,
            "O4'": -1.14,
            "C1'": 0.02,
            "C2'": 0.45,
            "C3'": 0.02,
            "O3'": -0.92,
            'N1': -1.66,
            'C2': 0.1,
            'O2': -0.58,
            'N3': 0.16,
            'C4': 0.25,
            'O4': -0.58,
            'C5': 0.1,
            'C6': 0.31,
            'C7': 0.45},
    'G': {'P': -0.94,
            'O1P': -0.7,
            'O2P': -0.22,
            "O5'": -0.5,
            "C5'": 0.45,
            "C4'": 0.02,
            "O4'": -1.14,
            "C1'": 0.02,
            "C2'": 0.02,
            "O2'": -0.92,
            "C3'": 0.02,
            "O3'": -0.92,
            'N9': -1.66,
            'C8': 0.31,
            'N7': -0.55,
            'C5': 0.25,
            'C6': 0.1,
            'O6': -0.58,
            'N1': -0.49,
            'C2': 0.1,
            'N2': -0.6,
            'N3': -0.07,
            'C4': -0.25},
    'A': {'P': -0.94,
            'O1P': -0.7,
            'O2P': -0.22,
            "O5'": -0.5,
            "C5'": 0.45,
            "C4'": 0.02,
            "O4'": -1.14,
            "C1'": 0.02,
            "C2'": 0.02,
            "O2'": -0.92,
            "C3'": 0.02,
            "O3'": -0.92,
            'N9': -1.66,
            'C8': 0.31,
            'N7': -0.55,
            'C5': 0.25,
            'C6': 0.1,
            'N6': -0.6,
            'N1': -0.49,
            'C2': 0.31,
            'N2': -0.6,
            'N3': -0.07,
            'C4': -0.25},
    'C': {'P': -0.94,
            'O1P': -0.7,
            'O2P': -0.22,
            "O5'": -0.5,
            "C5'": 0.45,
            "C4'": 0.02,
            "O4'": -1.14,
            "C1'": 0.02,
            "C2'": 0.02,
            "O2'": -0.92,
            "C3'": 0.02,
            "O3'": -0.92,
            'N1': -1.66,
            'C2': 0.1,
            'O2': -0.58,
            'N3': -0.29,
            'C4': 0.1,
            'N4': -0.6,
            'C5': 0.31,
            'C6': 0.31},
    'U': {'P': -0.94,
            'O1P': -0.7,
            'O2P': -0.22,
            "O5'": -0.5,
            "C5'": 0.45,
            "C4'": 0.02,
            "O4'": -1.14,
            "C1'": 0.02,
            "C2'": 0.02,
            "O2'": -0.92,
            "C3'": 0.02,
            "O3'": -0.92,
            'N1': -1.66,
            'C2': 0.1,
            'O2': -0.58,
            'N3': 0.16,
            'C4': 0.25,
            'O4': -0.58,
            'C5': 0.1,
            'C6': 0.31},
}


# Ordering is a mix between the AlphaFold ordering (for proteins) and the original spec 
# for nucleic acids from 1992: 
# https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
# rotation points for artificial 'wiggle' are: 5, 6, 12, 20, 25
# rotation point around the alpha carbon is: 2
atom_names = {
        # backbone atoms
        'N'  : 1, 
        'CA' : 2, # rotation point
        'C'  : 3, 
        'O'  : 4, 
        
        # sidechain atoms
        'CB' : 5, # rotation point
        'CG' : 6, # rotation point
        'CG1': 7, 
        'CG2': 8, 
        'OG' : 9, 
        'OG1': 10, 
        'SG' : 11, 
        'CD' : 12, # rotation point
        'CD1': 13, 
        'CD2': 14, 
        'ND1': 15, 
        'ND2': 16, 
        'OD1': 17, 
        'OD2': 18, 
        'SD' : 19, 
        'CE' : 20, # rotation point
        'CE1': 21, 
        'CE2': 23, 
        'CE3': 24,
        'NE' : 25, # rotation point
        'NE1': 26, 
        'NE2': 27, 
        'OE1': 28, 
        'OE2': 29, 
        'CH2': 30, 
        'NH1': 31, 
        'NH2': 32, 
        'OH' : 33, 
        'CZ' : 34, 
        'CZ2': 35,
        'CZ3': 36, 
        'NZ' : 37, 
        'OXT': 38, # terminus of a peptide chain when it ends in an oxygen, maybe move higher? (in the 'backbone')
        
        
        ## DNA 
        # currently works for RNA as well, but haven't optimised
        
        # phosphate    
        "P"  : 50, # connection to the previous ribose
        "O1P": 51,
        "02P": 52, 
        # ribose
        "O5'" : 53, 
        "C5'" : 54, 
        "C4'" : 55, # ring
        "O4'" : 56, # ring
        "C3'" : 57, # ring
        "O3'" : 58, # connection to the next phosphate
        "C2'" : 59, # ring
        "O2'" : 60, 
        "C1'" : 61, # ring # connection to the base
        
        # connection point to the base
        "N1" : 62,
        "N9" : 63,
        "N3" : 64
        
        # unsure how to proceed past this point, as atom names are reused inside of 
        # different bases but represent totally different locations like the N9 and N1 atoms 
        # can both be the connecting carbon to the ribose, or a carbon much further into the
        # structure see page 29: 
        # https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
}


# taken from biotite code: https://www.biotite-python.org/examples/gallery/structure/glycan_visualization.html
# originally adapted from "Mol*" Software
# The dictionary maps residue names of saccharides to their common names
SACCHARIDE_NAMES = {
    res_name : common_name for common_name, res_names in [
        ("Glc", ["GLC", "BGC", "Z8T", "TRE", "MLR"]),
        ("Man", ["MAN", "BMA"]),
        ("Gal", ["GLA", "GAL", "GZL", "GXL", "GIV"]),
        ("Gul", ["4GL", "GL0", "GUP", "Z8H"]),
        ("Alt", ["Z6H", "3MK", "SHD"]),
        ("All", ["AFD", "ALL", "WOO", "Z2D"]),
        ("Tal", ["ZEE", "A5C"]),
        ("Ido", ["ZCD", "Z0F", "4N2"]),
        ("GlcNAc", ["NDG", "NAG", "NGZ"]),
        ("ManNAc", ["BM3", "BM7"]),
        ("GalNAc", ["A2G", "NGA", "YYQ"]),
        ("GulNAc", ["LXB"]),
        ("AllNAc", ["NAA"]),
        ("IdoNAc", ["LXZ"]),
        ("GlcN", ["PA1", "GCS"]),
        ("ManN", ["95Z"]),
        ("GalN", ["X6X", "1GN"]),
        ("GlcA", ["GCU", "BDP"]),
        ("ManA", ["MAV", "BEM"]),
        ("GalA", ["ADA", "GTR", "GTK"]),
        ("GulA", ["LGU"]),
        ("TalA", ["X1X", "X0X"]),
        ("IdoA", ["IDR"]),
        ("Qui", ["G6D", "YYK"]),
        ("Rha", ["RAM", "RM4", "XXR"]),
        ("6dGul", ["66O"]),
        ("Fuc", ["FUC", "FUL", "FCA", "FCB"]),
        ("QuiNAc", ["Z9W"]),
        ("FucNAc", ["49T"]),
        ("Oli", ["DDA", "RAE", "Z5J"]),
        ("Tyv", ["TYV"]),
        ("Abe", ["ABE"]),
        ("Par", ["PZU"]),
        ("Dig", ["Z3U"]),
        ("Ara", ["64K", "ARA", "ARB", "AHR", "FUB", "BXY", "BXX"]),
        ("Lyx", ["LDY", "Z4W"]),
        ("Xyl", ["XYS", "XYP", "XYZ", "HSY", "LXC"]),
        ("Rib", ["YYM", "RIP", "RIB", "BDR", "0MK", "Z6J", "32O"]),
        ("Kdn", ["KDM", "KDN"]),
        ("Neu5Ac", ["SIA", "SLB"]),
        ("Neu5Gc", ["NGC", "NGE"]),
        ("LDManHep", ["GMH"]),
        ("Kdo", ["KDO"]),
        ("DDManHep", ["289"]),
        ("MurNAc", ["MUB", "AMU"]),
        ("Mur", ["1S4", "MUR"]),
        ("Api", ["XXM"]),
        ("Fru", ["BDF", "Z9N", "FRU", "LFR"]),
        ("Tag", ["T6T"]),
        ("Sor", ["SOE"]),
        ("Psi", ["PSV", "SF6", "SF9"]),
    ]
    for res_name in res_names
}

# currently not useing this, but potential for colouring / default colour glycans 
# or maybe also adding simplified representations of the glycans through GN
# currently just data storage for potential future projects
SACCHARIDE_REPRESENTATION = {
    "Glc": ("o", "royalblue"),
    "Man": ("o", "forestgreen"),
    "Gal": ("o", "gold"),
    "Gul": ("o", "darkorange"),
    "Alt": ("o", "pink"),
    "All": ("o", "purple"),
    "Tal": ("o", "lightsteelblue"),
    "Ido": ("o", "chocolate"),

    "GlcNAc": ("s", "royalblue"),
    "ManNAc": ("s", "forestgreen"),
    "GalNAc": ("s", "gold"),
    "GulNAc": ("s", "darkorange"),
    "AllNAc": ("s", "purple"),
    "IdoNAc": ("s", "chocolate"),

    "GlcN": ("1", "royalblue"),
    "ManN": ("1", "forestgreen"),
    "GalN": ("1", "gold"),

    "GlcA": ("v", "royalblue"),
    "ManA": ("v", "forestgreen"),
    "GalA": ("v", "gold"),
    "GulA": ("v", "darkorange"),
    "TalA": ("v", "lightsteelblue"),
    "IdoA": ("v", "chocolate"),

    "Qui": ("^", "royalblue"),
    "Rha": ("^", "forestgreen"),
    "6dGul": ("^", "darkorange"),
    "Fuc": ("^", "crimson"),

    "QuiNAc": ("P", "royalblue"),
    "FucNAc": ("P", "crimson"),

    "Oli": ("X", "royalblue"),
    "Tyv": ("X", "forestgreen"),
    "Abe": ("X", "darkorange"),
    "Par": ("X", "pink"),
    "Dig": ("X", "purple"),

    "Ara": ("*", "forestgreen"),
    "Lyx": ("*", "gold"),
    "Xyl": ("*", "darkorange"),
    "Rib": ("*", "pink"),

    "Kdn": ("D", "forestgreen"),
    "Neu5Ac": ("D", "mediumvioletred"),
    "Neu5Gc": ("D", "turquoise"),

    "LDManHep": ("H", "forestgreen"),
    "Kdo": ("H", "gold"),
    "DDManHep": ("H", "pink"),
    "MurNAc": ("H", "purple"),
    "Mur": ("H", "chocolate"),

    "Api": ("p", "royalblue"),
    "Fru": ("p", "forestgreen"),
    "Tag": ("p", "gold"),
    "Sor": ("p", "darkorange"),
    "Psi": ("p", "pink"),

    # Default representation
    None: ("h", "black")
}