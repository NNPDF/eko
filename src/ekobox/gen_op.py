def gen_op_card(Q2grid, hash, update=None):
    """
    Generates an operator card with some mandatory user choice and some
    default values which can be changed by the update input dict
    """
    # Constructing the dictionary with some default value (NB: ask if it relies on order)
    def_op = {  #'Q2grid': [100],
        "backward_inversion": "expanded",
        "debug_skip_non_singlet": False,
        "debug_skip_singlet": False,
        "ev_op_iterations": 1,
        "ev_op_max_order": 10,
        #'hash': 'ff080e6',
        "interpolation_is_log": True,
        "interpolation_polynomial_degree": 4,
        "interpolation_xgrid": [
            1e-07,
            1.6102620275609392e-07,
            2.592943797404667e-07,
            4.1753189365604003e-07,
            6.723357536499335e-07,
            1.0826367338740541e-06,
            1.7433288221999873e-06,
            2.8072162039411756e-06,
            4.520353656360241e-06,
            7.2789538439831465e-06,
            1.1721022975334793e-05,
            1.8873918221350995e-05,
            3.039195382313195e-05,
            4.893900918477499e-05,
            7.880462815669905e-05,
            0.0001268961003167922,
            0.00020433597178569417,
            0.00032903445623126676,
            0.0005298316906283707,
            0.0008531678524172806,
            0.0013738237958832637,
            0.00221221629107045,
            0.003562247890262444,
            0.005736152510448681,
            0.009236708571873866,
            0.014873521072935119,
            0.02395026619987486,
            0.03856620421163472,
            0.06210169418915616,
            0.1,
            0.1473684210526316,
            0.19473684210526315,
            0.24210526315789474,
            0.2894736842105263,
            0.33684210526315794,
            0.38421052631578945,
            0.43157894736842106,
            0.4789473684210527,
            0.5263157894736842,
            0.5736842105263158,
            0.6210526315789474,
            0.6684210526315789,
            0.7157894736842105,
            0.7631578947368421,
            0.8105263157894737,
            0.8578947368421053,
            0.9052631578947369,
            0.9526315789473684,
            1.0,
        ],
    }
    # Adding the mandatory inputs
    def_op["Q2grid"] = Q2grid
    def_op["hash"] = hash
    # Update user choice (NB: Allow also new entries?)
    if isinstance(update, dict):
        for k in update.keys():
            if k not in def_op.keys():
                raise ValueError("Provided key not in operators card")
        def_op.update(update)
    return def_op
