# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (03/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
#              Siyuan Niu  (<siyuan.niu@lirmm.fr>)
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================

import logging
import random
import typing as ty
from copy import deepcopy
from math import exp

logger = logging.getLogger("hamap.optimisation.simulated_annealing")

StateType = ty.NewType("StateType", ty.Any)


def _accept_neighbour(
    neighbour_cost: float, current_cost: float, temperature: float
) -> bool:
    try:
        p = exp(-(neighbour_cost - current_cost) / temperature)
    except OverflowError:
        return True
    return p >= random.random()


def simulated_annealing(
    initial_state: StateType,
    cost_function: ty.Callable[[StateType], float],
    get_neighbour: ty.Callable[[StateType], StateType],
    initial_temperature: float,
    max_iterations: int,
    schedule_function: ty.Callable[[float], float],
    atol: float = 0.0,
    stop_temperature: float = 1e-6,
) -> ty.Tuple[StateType, float, int]:
    """Optimise using simulated annealing procedure.

    :param initial_state: the initial state of the procedure.
    :param cost_function: a function computing the cost of a state.
    :param get_neighbour: a function returning a neighbour of a given state.
    :param initial_temperature: the initial temperature of the annealing process.
    :param max_iterations: the maximum number of iterations the annealing procedure
        is allowed to perform before stopping. This is a maximum, the annealing
        procedure might stop before reaching this number of iteration because of a
        cost that is under target_cost.
    :param schedule_function: a function that will take the current temperature and will
        return an updated temperature according to some policy. If you want the
        annealing procedure to be efficient, this function should be decreasing (i.e.
        never return a temperature higher than the one given as input) and should
        become small when the number of iterations starts to approach the maximum
        number of allowed iterations.
    :param atol: a stop criterion that takes into account the current cost. If
        the cost of the current state is lower than this threshold value,
        the optimisation will stop and return the current state.
    :param stop_temperature: a stop criterion for the temperature. If the current
        temperature is lower than this value, the algorithm stops.
    :return:
    """

    if max_iterations < 1:
        raise RuntimeError(
            "Number of iterations should be 1 or more for simulated annealing."
        )

    current_state = initial_state
    best_state = deepcopy(current_state)
    current_temperature = initial_temperature
    current_cost = cost_function(current_state)
    best_cost = current_cost

    for i in range(max_iterations):
        neighbour = get_neighbour(current_state)
        neighbour_cost = cost_function(neighbour)

        if _accept_neighbour(neighbour_cost, current_cost, current_temperature):
            current_state = neighbour
            current_cost = neighbour_cost

        if current_cost < best_cost:
            best_state = deepcopy(current_state)
            best_cost = current_cost

        if best_cost < atol or current_temperature < stop_temperature:
            break
        current_temperature = schedule_function(current_temperature)

    if current_temperature > 0.1 and best_cost > atol:
        logger.warning(
            "Temperature at the end of simulated annealing is still too "
            f"high ({current_temperature:.2f}). Consider reducing the initial "
            f"temperature or changing the schedule function to a more rapidly "
            f"decreasing one. {max_iterations}"
        )
    return best_state, best_cost, i
