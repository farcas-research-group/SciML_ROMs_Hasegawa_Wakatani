import tecplot as tec


# read one available tecplot file to have access ot the geometry
def get_tecplot_setup(
    mesh_loc,
    available_datafile,
    addtl_vars,
    ignore_vars=False,
    vars_to_ignore=None,
    out_dtype=2,
):
    data_set = tec.data.load_tecplot_szl(
        [mesh_loc, available_datafile], read_data_option=2
    )
    t_zone = data_set.zone(0)
    n_dims = t_zone.rank

    # t_zone.dataset.variable("Density [[]kg/m^3[]]").name = 'Density'
    # t_zone.dataset.variable("Pressure [[]Pa[]]").name = 'Pressure'
    # t_zone.dataset.variable("Temperature [[]K[]]").name = 'Temperature'
    # t_zone.dataset.variable("Subgrid Kinetic Energy [[]m^2/s^2[]]").name = 'Subgrid Kinetic Energy'
    # t_zone.dataset.variable("U-Velocity [[]m/s[]]").name = 'U-Velocity'
    # t_zone.dataset.variable("V-Velocity [[]m/s[]]").name = 'V-Velocity'
    # t_zone.dataset.variable("W-Velocity [[]m/s[]]").name = 'W-Velocity'

    if ignore_vars:
        varList_cc = data_set.variable_names[n_dims:]
        varIDs_cc = []

        for var in varList_cc:
            varIDs_cc.append(data_set.variable(var))

        data_set.delete_variables(varIDs_cc)

        for var in varList_cc:
            data_set.add_variable(
                var,
                dtypes=tec.constant.FieldDataType(out_dtype),
                locations=tec.constant.ValueLocation(0),
            )

    # make Tecplot fields for new data fields
    for var in addtl_vars:
        data_set.add_variable(
            var,
            dtypes=tec.constant.FieldDataType(out_dtype),
            locations=tec.constant.ValueLocation(0),
        )

    return data_set, t_zone, n_dims


# write data to tecplot file
def write_data_to_szplt_file(
    varList_cc, data_set, ex_zone, n_dims, rom_data, t, dt, out_file
):
    print("inside write data to szplt")
    print(varList_cc)

    print("rom data shape", rom_data.shape[1])

    assert len(varList_cc) == rom_data.shape[1]

    # write data to Tecplot zone
    for iter_var, var in enumerate(varList_cc):
        # pdb.set_trace()
        ex_zone.values(var)[:] = rom_data[:, iter_var]

    # update solution time
    ex_zone.solution_time = t * dt

    tec.data.save_tecplot_szl(out_file, dataset=data_set)
