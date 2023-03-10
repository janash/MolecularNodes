import bpy
import numpy as np
from . import data
from . import coll
from .load import create_object, add_attribute
import warnings


def load_trajectory(file_top, 
                    file_traj,
                    md_start = 1, 
                    md_end = 50, 
                    md_step = 1,
                    world_scale = 0.01, 
                    include_bonds = False, 
                    del_solvent = False,
                    solute ='element Li',
                    solvent_names ='EA,FEC,PF6',
                    solvent_groups ='resid 1-235,resid 235-600,byres element P',
                    solute_index=603,
                    frame_index=0,
                    name = "default"
                    ):
    
    import MDAnalysis as mda
    import MDAnalysis.transformations as trans

    
    # initially load in the trajectory
    if file_traj == "":
        univ = mda.Universe(file_top)
    else:
        univ = mda.Universe(file_top, file_traj)
        
    # separate the trajectory, separate to the topology or the subsequence selections
    traj = univ.trajectory[md_start:md_end:md_step]
    

    
    if solute != "":
        try:
            from solvation_analysis.solute import Solute

            solute_real = univ.select_atoms(solute)

            list_solv_groups = solvent_groups.split(",")
            list_solv_name= solvent_names.split(",")

            solv_resid=[]
            for i in list_solv_groups:
                solv_resid.append(univ.select_atoms(i))

            solv_dict=dict(zip(list_solv_name,solv_resid))

            # instantiate solution
            solute_obj= Solute.from_atoms(solute_real,
                                solv_dict)
            
            solute_obj.run()
            shell = solute_obj.get_shell(solute_index, frame_index)
            univ=shell
            print(univ)

        except:
            warnings.warn(f"Unable to apply selection: '{solute}'. Loading entire topology.")


    
    # Try and extract the elements from the topology. If the universe doesn't contain
    # the element information, then guess based on the atom names in the toplogy
    try:
        elements = univ.atoms.elements.tolist()
    except:
        try:
            elements = [mda.topology.guessers.guess_atom_element(x) for x in univ.atoms.names]
        except:
            pass
        


    if hasattr(univ, 'bonds') and include_bonds:
    # If there is a selection, we need to recalculate the bond indices
        if solute !="":
            index_map = { index:i for i, index in enumerate(univ.atoms.indices) }
            new_bonds = []
            for bond in univ.bonds.indices:
                try:
                    new_index = [index_map[y] for y in bond]
                    new_bonds.append(new_index)
                except KeyError:
                    # fragment - one of the atoms in the bonds was 
                    # deleted by the selection, so we shouldn't 
                    # pass this as a bond.  
                    pass
                
            bonds = np.array(new_bonds)
        else:
            bonds = univ.bonds.indices

    else:
        bonds = []



    # create the initial model
    mol_object = create_object(
        name = name,
        collection = coll.mn(),
        locations = univ.atoms.positions * world_scale, 
        bonds = bonds
    )
    
    ## add the attributes for the model

    # The attributes for the model are initially defined as single-use functions. This allows
    # for a loop that attempts to add each attibute by calling the function. Only during this
    # loop will the call fail if the attribute isn't accessible, and the warning is reported
    # there rather than setting up a try: except: for each individual attribute which makes
    # some really messy code.
    
    def att_atomic_number():
        atomic_number = np.array(list(map(
            # if getting the element fails for some reason, return an atomic number of -1
            lambda x: data.elements.get(x, {"atomic_number": -1}).get("atomic_number"), 
            np.char.title(elements) 
        )))
        return atomic_number

    def att_vdw_radii():
        try:
            vdw_radii = np.array(list(map(
                lambda x: mda.topology.tables.vdwradii.get(x, 1), 
                np.char.upper(elements)
            )))
        except:
            # if fail to get radii, just return radii of 1 for everything as a backup
            vdw_radii = np.ones(len(univ.atoms.names))
            warnings.warn("Unable to extract VDW Radii. Defaulting to 1 for all points.")
        
        return vdw_radii * world_scale
    
    def att_res_id():
        return univ.atoms.resnums
    
    def att_res_name():
        res_names =  np.array(list(map(lambda x: x[0: 3], univ.atoms.resnames)))
        res_numbers = np.array(list(map(
            lambda x: data.residues.get(x, {'res_name_num': 0}).get('res_name_num'), 
            res_names
            )))
        return res_numbers
    
    def att_b_factor():
        return univ.atoms.tempfactors
    
    def att_chain_id():
        chain_id = univ.atoms.chainIDs
        chain_id_unique = np.unique(chain_id)
        chain_id_num = np.array(list(map(lambda x: np.where(x == chain_id_unique)[0][0], chain_id)))
        mol_object['chain_id_unique'] = chain_id_unique
        return chain_id_num
    
    # returns a numpy array of booleans for each atom, whether or not they are in that selection
    def bool_selection(selection):
        return np.isin(univ.atoms.ix, univ.select_atoms(selection).ix).astype(bool)
    
    def att_is_backbone():
        return bool_selection("backbone or nucleicbackbone")
    
    def att_is_alpha_carbon():
        return bool_selection('name CA')
    
    def att_is_solvent():
        return bool_selection('name OW or name HW1 or name HW2')
    
    def att_atom_type():
        return np.array(univ.atoms.types, dtype = int)
    
    def att_is_nucleic():
        return bool_selection('nucleic')
    
    def att_is_peptide():
        return bool_selection('protein')

    attributes = (
        {'name': 'atomic_number',   'value': att_atomic_number,   'type': 'INT',     'domain': 'POINT'}, 
        {'name': 'vdw_radii',       'value': att_vdw_radii,       'type': 'FLOAT',   'domain': 'POINT'},
        {'name': 'res_id',          'value': att_res_id,          'type': 'INT',     'domain': 'POINT'}, 
        {'name': 'res_name',        'value': att_res_name,        'type': 'INT',     'domain': 'POINT'}, 
        {'name': 'b_factor',        'value': att_b_factor,        'type': 'float',   'domain': 'POINT'}, 
        {'name': 'chain_id',        'value': att_chain_id,        'type': 'INT',     'domain': 'POINT'}, 
        {'name': 'atom_types',      'value': att_atom_type,       'type': 'INT',     'domain': 'POINT'}, 
        {'name': 'is_backbone',     'value': att_is_backbone,     'type': 'BOOLEAN', 'domain': 'POINT'}, 
        {'name': 'is_alpha_carbon', 'value': att_is_alpha_carbon, 'type': 'BOOLEAN', 'domain': 'POINT'}, 
        {'name': 'is_solvent',      'value': att_is_solvent,      'type': 'BOOLEAN', 'domain': 'POINT'}, 
        {'name': 'is_nucleic',      'value': att_is_nucleic,      'type': 'BOOLEAN', 'domain': 'POINT'}, 
        {'name': 'is_peptide',      'value': att_is_peptide,      'type': 'BOOLEAN', 'domain': 'POINT'}, 
    )
    
    for att in attributes:
        # tries to add the attribute to the mesh by calling the 'value' function which returns
        # the required values do be added to the domain.
        try:
            add_attribute(mol_object, att['name'], att['value'](), att['type'], att['domain'])
        except:
            warnings.warn(f"Unable to add attribute: {att['name']}.")

    # add the custom selections if they exist
    custom_selections = bpy.context.scene.trajectory_selection_list
    if custom_selections:
        for sel in custom_selections:
            try:
                add_attribute(
                    object=mol_object, 
                    name=sel.name, 
                    data=bool_selection(sel.selection), 
                    type = "BOOLEAN", 
                    domain = "POINT"
                    )
            except:
                warnings.warn("Unable to add custom selection: {}".format(sel.name))

    coll_frames = coll.frames(name)
    
    add_occupancy = True
    for ts in traj:
        frame = create_object(
            name = name + "_frame_" + str(ts.frame),
            collection = coll_frames, 
            locations = univ.atoms.positions * world_scale
        )
        # adds occupancy data to each frame if it exists
        # This is mostly for people who want to store frame-specific information in the 
        # b_factor but currently neither biotite nor MDAnalysis give access to frame-specific
        # b_factor information. MDAnalysis gives frame-specific access to the `occupancy` 
        # so currently this is the only method to get frame-specific data into MN
        # for more details: https://github.com/BradyAJohnston/MolecularNodes/issues/128
        if add_occupancy:
            try:
                add_attribute(frame, 'occupancy', ts.data['occupancy'])
            except:
                add_occupancy = False
    
    # disable the frames collection from the viewer
    bpy.context.view_layer.layer_collection.children[coll.mn().name].children[coll_frames.name].exclude = True
    
    return mol_object, coll_frames


