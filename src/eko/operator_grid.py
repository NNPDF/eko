"""
    This module contains the OperatorGrid class
    q inside this class refers always to q^{2}
"""

import numpy as np
from eko.operator import Operator, OperatorMember
import eko.utils as utils
import logging
logger = logging.getLogger(__name__)
# evolution basis names
Vs = ["V3", "V8", "V15", "V24", "V35"]
Ts = ["T3", "T8", "T15", "T24", "T35"]

class OperatorMaster:
    """
        The OperatorMaster is instantiated for a given set of parameters
        And informs the generation of operators
    """

    def __init__(self, alpha_generator, kernel_dispatcher, xgrid, nf):
        # Get all the integrands necessary for singlet and not singlet for nf
        self._kernel_dispatcher = kernel_dispatcher
        self._alpha_gen = alpha_generator
        self._xgrid = xgrid
        self._nf = nf
        self._integrands_ns = None
        self._integrands_s = None

    def _compile(self):
        self._integrands_ns = self._kernel_dispatcher.get_non_singlet_for_nf(self._nf)
        self._integrands_s = self._kernel_dispatcher.get_singlet_for_nf(self._nf)

    def get_op(self, q_from, q_to, generate = False):
        if self._integrands_s is None or self._integrands_ns is None:
            self._compile()
        # Generate the metadata for this operator
        metadata = {
                'q' : q_to,
                'qref' : q_from,
                'nf' : self._nf
                }
        # Generate the necessary parameters to compute the operator
        delta_t = self._alpha_gen.delta_t(q_from, q_to)
        op = Operator(delta_t, self._xgrid, self._integrands_ns, self._integrands_s, metadata)
        if generate:
            op.compute()
        return op


class OperatorGrid:
    """
        The operator grid is the driver class of the evolution.
        It receives as input a threshold holder and a generator of alpha_s

        From that point onwards it can compute any operator at any q

        Parameters
        ----------
            threshold_holder: eko.thresholds.Threshold
                Instance of the Threshold class containing information about the thresholds
            alpha_generator: eko.alpha_s.StrongCoupling
                Instance of the StrongCoupling class able to generate a_s for any q
    """

    def __init__(self, threshold_holder, alpha_generator, kernel_dispatcher, xgrid):
        self._threshold_holder = threshold_holder
        self._op_masters = {}
        for nf in threshold_holder.nf_range():
            # Compile the kernels for each nf
            kernel_dispatcher.set_up_all_integrands(nf)
            # Set up the OP Master for each nf
            self._op_masters[nf] = OperatorMaster(alpha_generator, kernel_dispatcher, xgrid, nf)
        self._alpha_gen = alpha_generator
        self._kernels = kernel_dispatcher
        self._threshold_operators = {}
        self._op_grid = {}
        self.qmax = -1
        self.qmin = np.inf

    def _generate_thresholds_op(self, area_list):
        """ Generate the threshold operators """
        # Get unique areas
        q_from = self._threshold_holder.qref
        nf = self._threshold_holder.nf_ref
        for area in area_list:
            q_to = area.qref
            if q_to == q_from:
                continue
            new_op = (q_from, q_to)
            if new_op not in self._threshold_operators:
                self._threshold_operators[new_op] = self._op_masters[nf].get_op(q_from, q_to, generate=True)

            nf = area.nf
            q_from = q_to

    def _get_jumps(self, q):
        """ Receives a value of q and generates a list of operators to multiply for in order to get
        down to q0 """
        full_area_path = self._threshold_holder.get_path_from_q0(q)
        # The last one is where q resides so it is not needed
        area_path = full_area_path[:-1]
        op_list = []
        for area in area_path:
            q_from = area.qref
            q_to = area.q_towards(q)
            op_list.append(self._threshold_operators[(q_from, q_to)])
        return op_list

    def set_q_limits(self, qmin, qmax):
        """ Sets up the limits of the grid in q^2 to be computed by the OperatorGrid

        This function computes the necessary operators to go between areas

        Parameters
        ----------
            qmin: float
                Minimum value of q that will be computed
            qmax: float
                Maximum value of q that will be computed
        """
        if qmin <= 0.0:
            raise ValueError(f"Values of q below 0.0 are not accepted, received {qmin}")
        if qmin > qmax:
            raise ValueError(f"Minimum q is above maximum q (error: {qmax} < {qmin})")
        # Get the path from q0 to qmin and qmax
        from_qmin = self._threshold_holder.get_path_from_q0(qmin)
        from_qmax = self._threshold_holder.get_path_from_q0(qmax)
        self._generate_thresholds_op(from_qmin)
        self._generate_thresholds_op(from_qmax)

    def _compute_raw_grid(self, qgrid):
        """ Receives a grid in q^2 and computes each opeator inside its
        area with reference value the q_ref of its area

        Parameters
        ----------
            qgrid: list
                List of q^2
        """
        area_list = self._threshold_holder.get_areas(qgrid)
        for area, q in zip(area_list, qgrid):
            q_from = area.qref
            nf = area.nf
            self._op_grid[q] = self._op_masters[nf].get_op(q_from, q)
        # Now perform the computation, TODO everything in parallel
        for _, op in self._op_grid.items():
            op.compute()

    def compute_qgrid(self, qgrid):
        """ Receives a grid in q^2 and computes all operations necessary
        to return any operator at any given q for the evolution between qref and qgrid

        Parameters
        ----------
            qgrid: list
                List of q^2
        """
        if isinstance(qgrid, (np.float, np.int, np.integer)):
            qgrid = [qgrid]
        # Check max and min of the grid and reset the limits if necessary
        qmax = np.max(qgrid)
        qmin = np.min(qgrid)
        self.set_q_limits(qmin, qmax)
        # Now compute all raw operators
        self._compute_raw_grid(qgrid)

    def get_op_at_Q(self, q):
        """
            Return the operator at Q
        """
        # Check the path to q0 for this operator
        if q in self._op_grid:
            operator = self._op_grid[q]
        else:
            self.compute_qgrid(q)
            logger.warning("Q=%f not found in the grid, computing...", q)
            operator = self._op_grid[q]
        qref = operator.qref
        # Check the path the operator has to go through
        operators_to_q0 = self._get_jumps(qref)
        # TODO: do this in a more elegant way
        number_of_thresholds = len(operators_to_q0)
        if number_of_thresholds == 0:
            return operator.pdf_space(self._threshold_holder._scheme)

        number_of_thresholds = 2

        # If we have to go through some threshold, prepare the operations
        nf_init = operator.nf - number_of_thresholds
        # Operators to multiply
        op_to_multiplty = [i._internal_ret for i in reversed(operators_to_q0 + [operator])]
        op_to_multiply = [i._internal_ops for i in reversed(operators_to_q0 + [operator])]
        return_dictionary = {}

        # TODO everything here should be written much more concise

        ret = {"operators": {}, "operator_errors": {}}
        def good_helper(name, paths):
            new_op = utils.operator_product(op_to_multiply, paths)
            new_op.name = name
            return new_op
        
        def set_helper(name, path):
            path_setup[name] = path
            return None




        # Generic fill up
        nfmo = nf_init - 1
        an = number_of_thresholds+1 # Number of areas to pass
        anmo = number_of_thresholds
        path_setup = {
                "V.V"           : [an*["NS_v"]], # V = \mul v
                f"{Vs[nfmo]}.V" : [anmo*["NS_m"] + ["NS_v"]], # (\mul -)*v = V15
                f"{Ts[nfmo]}.S" : [anmo*["NS_p"] + ["S_qq"]], # (\mul +)*v = T15_u
                f"{Ts[nfmo]}.g" : [anmo*["NS_p"] + ["S_qg"]], # (\mul +)*v = T15_d
                }

        # Dynamical fill up
        # (\mul -) = V3, V8
        for b in Vs[:nfmo]:
            path_setup[f"{b}.{b}"] = [an*["NS_m"]]
        # (\mul +) = T3, T8
        for b in Ts[:nfmo]:
            path_setup[f"{b}.{b}"] = [an*["NS_p"]]

        # Generate complicated paths
        paths_qq = utils.get_singlet_paths("q", "q", an)
        paths_qg = utils.get_singlet_paths("q", "g", an)

        # Singlet + gluon
        path_setup["S.S"]= paths_qq
        path_setup["S.g"]= paths_qg
        path_setup["g.S"]= utils.get_singlet_paths("g", "q", an)
        path_setup["g.g"]= utils.get_singlet_paths("g", "g", an)


        if number_of_thresholds == 1:
            # Setup all final results and their respective arrival paths
#             nfmo = nf_init - 1
#             path_setup = {
#                     "V.V": [["NS_v", "NS_v"]], # v.v = V
#                     f"{Vs[nfmo]}.V" : [["NS_m", "NS_v"]], # -.v
#                     f"{Ts[nfmo]}.S" : [["NS_p", "S_qq"]], # +.S
#                     f"{Ts[nfmo]}.g" : [["NS_p", "S_qg"]], # +.S
#                     }
            # Dynamical fill up
            # -.-
#             for b in Vs[:nfmo]:
#                 path_setup[f"{b}.{b}"] = [["NS_m", "NS_m"]]
            # v.v for higher combinations
            for b in Vs[nf_init:]:
                path_setup[f"{b}.V"] = [["NS_v", "NS_v"]]
            # +.+
#             for b in Ts[: nf_init - 1]:
#                 path_setup[f"{b}.{b}"] = [["NS_p", "NS_p"]]
            # S.S
            paths_qq = utils.get_singlet_paths("q", "q", 2)
            paths_qg = utils.get_singlet_paths("q", "g", 2)
            for b in Ts[nf_init:]:
                path_setup[f"{b}.S"] = paths_qq
                path_setup[f"{b}.g"] = paths_qg
            # Singlet + gluon
#             path_setup["S.S"]= paths_qq
#             path_setup["S.g"]= paths_qg
#             path_setup["g.S"]= utils.get_singlet_paths("g", "q", 2)
#             path_setup["g.g"]= utils.get_singlet_paths("g", "g", 2)

            for operator_path in self._threshold_holder._operator_paths:
                where = operator_path.name
                paths = operator_path.get_path(operator.nf, number_of_thresholds)
                for origin, path in paths.items():
                    key = f'{where}.{origin}'
                    print(key)
                    print(f"ORIGINAL: {path_setup[key]}")
                    print(f"COMPUTED: {path}")
                    print( path == path_setup[key] )

            import ipdb
            ipdb.set_trace()

            print("---------------------------------------")
        elif number_of_thresholds == 2:
            path_setup = {}
            # join quarks flavors
            # v.v.v = V
            set_helper("V.V", [["NS_v", "NS_v", "NS_v"]])
            # -.-.-
            for v in Vs[: nf_init - 1]:
                set_helper(f"{v}.{v}", [["NS_m", "NS_m", "NS_m"]])
            # -.-.v
            b = Vs[nf_init - 1]
            set_helper(f"{b}.V", [["NS_m", "NS_m", "NS_v"]])
            # -.v.v = V24
            b = Vs[nf_init]
            set_helper(f"{b}.V", [["NS_m", "NS_v", "NS_v"]])
            # v.v.v for higher combinations
            for b in Vs[nf_init + 1 :]:
                set_helper(f"{b}.V", [["NS_v", "NS_v", "NS_v"]]) # TODO
            # +.+.+
            for b in Ts[: nf_init - 1]:
                set_helper(f"{b}.{b}", [["NS_p", "NS_p", "NS_p"]])
            # +.+.S
            b = Ts[nf_init - 1]
            set_helper(f"{b}.S", [["NS_p", "NS_p", "S_qq"]])
            # +.S.S
            b = Ts[nf_init]
            paths_qq_2 = utils.get_singlet_paths("q", "q", 2)
            for p in paths_qq_2:
                p.insert(0, "NS_p")
            set_helper(f"{b}.S", paths_qq_2)
            # S.S.S
            paths_qq_3 = utils.get_singlet_paths("q", "q", 3)
            paths_qg_3 = utils.get_singlet_paths("q", "g", 3)
            for b in Ts[nf_init + 1 :]:
                set_helper(f"{b}.S", paths_qq_3)
                set_helper(f"{b}.g", paths_qg_3)

            # Singlet + gluon
            set_helper("S.S", paths_qq_3)
            set_helper("S.g", paths_qg_3)
            set_helper("g.S", utils.get_singlet_paths("g", "q", 3))
            set_helper("g.g", utils.get_singlet_paths("g", "g", 3))

            # Problems with
            probems = ["V15.V", "T15.S", "T15.g",
                    "T24.g", "T35.g"]
            #depth is wrong also here
            errors = ["T24.S", 
                "T35.S",]


            print("Initiating test")
            for operator_path in self._threshold_holder._operator_paths:
                where = operator_path.name
                paths = operator_path.get_path(operator.nf, number_of_thresholds)
                for origin, path in paths.items():
                    key = f'{where}.{origin}'
                    print(key)
                    try:
                        print(f"ORIGINAL: {path_setup[key]}")
                        print(f"COMPUTED: {path}")
                        print(path == path_setup[key])
                    except:
                        print(f" > > Problem with {key}")
            print("Finalising test")

            import ipdb
            ipdb.set_trace()


            return ret
        elif number_of_thresholds == 3:
            path_setup = {}
            # join quarks flavors
            # v.v.v.v = V
            set_helper("V.V", [["NS_v", "NS_v", "NS_v", "NS_v"]])
            # -.-.-.- = V3,V8
            for v in Vs[:2]:
                set_helper(f"{v}.{v}", [["NS_m", "NS_m", "NS_m", "NS_m"]])
            # -.-.-.v = V15
            b = Vs[2]
            set_helper(f"{b}.V", [["NS_m", "NS_m", "NS_m", "NS_v"]])
            # -.-.v.v = V24
            b = Vs[4]
            set_helper(f"{b}.V", [["NS_m", "NS_m", "NS_v", "NS_v"]])
            # -.v.v.v = V35
            b = Vs[5]
            set_helper(f"{b}.V", [["NS_m", "NS_v", "NS_v", "NS_v"]])
            # +.+.+.+ = T3,T8
            for b in Ts[:2]:
                set_helper(f"{b}.{b}", [["NS_p", "NS_p", "NS_p", "NS_p"]])
            # +.+.+.S = T15
            b = Ts[2]
            set_helper(f"{b}.S", [["NS_p", "NS_p", "NS_p", "S_qq"]])
            # +.+.S.S = T24
            b = Ts[3]
            paths_qq_2 = utils.get_singlet_paths("q", "q", 2)
            for p in paths_qq_2:
                p.insert(0, "NS_p")
                p.insert(0, "NS_p")
            set_helper(f"{b}.S", paths_qq_2)
            # +.S.S.S = T35
            b = Ts[4]
            paths_qq_3 = utils.get_singlet_paths("q", "q", 3)
            for p in paths_qq_3:
                p.insert(0, "NS_p")
            set_helper(f"{b}.S", paths_qq_3)

            # Singlet + gluon
            set_helper("S.S", utils.get_singlet_paths("q", "q", 4))
            set_helper("S.g", utils.get_singlet_paths("q", "g", 4))
            set_helper("g.S", utils.get_singlet_paths("g", "q", 4))
            set_helper("g.g", utils.get_singlet_paths("g", "g", 4))

            for operator_path in self._threshold_holder._operator_paths:
                where = operator_path.name
                paths = operator_path.get_path(operator.nf, number_of_thresholds)
                for origin, path in paths.items():
                    key = f'{where}.{origin}'
                    print(key)
                    print(f"ORIGINAL: {path_setup[key]}")
                    print(f"COMPUTED: {path}")
                    print( path == path_setup[key] )

            import ipdb
            ipdb.set_trace()
            return ret

            for operator_path in self._threshold_holder._operator_paths:
                where = operator_path.name
                paths = operator_path.get_path(operator.nf, number_of_thresholds)
                for origin, path in paths.items():
                    key = f'{where}.{origin}'
                    print(f"ORIGINAL: {path_setup[key]}")
                    print(f"COMPUTED: {path}")
                    print(path == path_setup[key])

        for operator_path in self._threshold_holder._operator_paths:
            where = operator_path.name
            paths = operator_path.get_path(operator.nf, number_of_thresholds)
            for origin, path in paths.items():
                name = f'{where}.{origin}'
                return_dictionary[name] = good_helper(name, path)

#         # Now create the results
#         for name, path in path_setup.items():
#             return_dictionary[name] = good_helper(name, path)

        for key, item in return_dictionary.items():
            ret["operators"][key] = item.value
            ret["operator_errors"][key] = item.error

        return ret
