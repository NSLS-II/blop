import numpy as np
import random

import bluesky.plan_stubs as bps
import bluesky.plans as bp

from .de_opt_utils import check_opt_bounds, move_to_optimized_positions


def omea_evaluation(motors, bounds, popsize, num_interm_vals, num_scans_at_once,
                    uids, flyer_name, intensity_name, db):
    if motors is not None:
        # hardware
        # get the data from databroker
        current_fly_data = []
        pop_intensity = []
        pop_positions = []
        max_intensities = []
        max_int_pos = []
        for uid in uids:
            current_fly_data.append(db[uid].table(flyer_name))
        for i, t in enumerate(current_fly_data):
            pop_pos_dict = {}
            positions_dict = {}
            max_int_index = t[f'{flyer_name}_{intensity_name}'].idxmax()
            for elem, param in motors.items():
                positions_dict[elem] = {}
                pop_pos_dict[elem] = {}
                for param_name in param.keys():
                    positions_dict[elem][param_name] = t[f'{flyer_name}_{elem}_{param_name}'][max_int_index]
                    pop_pos_dict[elem][param_name] = t[f'{flyer_name}_{elem}_{param_name}'][len(t)]
            pop_intensity.append(t[f'{flyer_name}_{intensity_name}'][len(t)])
            max_intensities.append(t[f'{flyer_name}_{intensity_name}'][max_int_index])
            pop_positions.append(pop_pos_dict)
            max_int_pos.append(positions_dict)
        # compare max of each fly scan to population
        # replace population/intensity with higher vals, if they exist
        for i in range(len(max_intensities)):
            if max_intensities[i] > pop_intensity[i]:
                pop_intensity[i] = max_intensities[i]
                for elem, param in max_int_pos[i].items():
                    for param_name, pos in param.items():
                        pop_positions[i][elem][param_name] = pos
        return pop_positions, pop_intensity
    elif bounds is not None and popsize is not None and num_interm_vals is \
            not None and num_scans_at_once is not None and motors is None:
        # sirepo simulation
        pop_positions = []
        pop_intensities = []
        # get data from databroker
        fly_data = []
        # for i in range(-int(num_records), 0):
        for uid in uids:
            fly_data.append(db[uid].table(flyer_name))
        interm_pos = []
        interm_int = []
        # Create all sets of indices for population values first
        pop_indxs = [[0, 1]]
        while len(pop_indxs) < popsize:
            i_index = pop_indxs[-1][0]
            j_index = pop_indxs[-1][1]
            pre_mod_val = j_index + num_interm_vals + 1
            mod_res = pre_mod_val % num_scans_at_once
            int_div_res = pre_mod_val // num_scans_at_once
            if mod_res == 0:
                i_index = i_index + (int_div_res - 1)
                j_index = pre_mod_val
            else:
                i_index = i_index + int_div_res
                j_index = mod_res
            pop_indxs.append([i_index, j_index])
        curr_pop_index = 0
        for i in range(len(fly_data)):
            curr_interm_pos = []
            curr_interm_int = []
            for j in range(1, len(fly_data[i]) + 1):
                if (i == pop_indxs[curr_pop_index][0] and
                        j == pop_indxs[curr_pop_index][1]):
                    pop_intensities.append(fly_data[i][f'{flyer_name}_{intensity_name}'][j])
                    indv = {}
                    for elem, param in bounds.items():
                        indv[elem] = {}
                        for param_name in param.keys():
                            indv[elem][param_name] = fly_data[i][f'{flyer_name}_{elem}_{param_name}'][j]
                    pop_positions.append(indv)
                    curr_pop_index += 1
                else:
                    curr_interm_int.append(fly_data[i][f'{flyer_name}_{intensity_name}'][j])
                    indv = {}
                    for elem, param in bounds.items():
                        indv[elem] = {}
                        for param_name in param.keys():
                            indv[elem][param_name] = fly_data[i][f'{flyer_name}_{elem}_{param_name}'][j]
                    curr_interm_pos.append(indv)
            interm_pos.append(curr_interm_pos)
            interm_int.append(curr_interm_int)
        # picking best positions
        interm_max_idx = []
        for i in range(len(interm_int)):
            if len(interm_int[i]) == 0:
                interm_max_idx.append(None)
            else:
                curr_max_int = np.max(interm_int[i])
                interm_max_idx.append(interm_int[i].index(curr_max_int))
        for i in range(len(interm_max_idx)):
            if interm_max_idx[i] is None:
                pass
            else:
                if interm_int[i][interm_max_idx[i]] > pop_intensities[i + 1]:
                    pop_intensities[i + 1] = interm_int[i][interm_max_idx[i]]
                    pop_positions[i + 1] = interm_pos[i][interm_max_idx[i]]
        return pop_positions, pop_intensities


def ensure_bounds(vec, bounds):
    # Makes sure each individual stays within bounds and adjusts them if they aren't
    vec_new = {}
    # cycle through each variable in vector
    for elem, param in vec.items():
        vec_new[elem] = {}
        for param_name, pos in param.items():
            # variable exceeds the minimum boundary
            if pos < bounds[elem][param_name][0]:
                vec_new[elem][param_name] = bounds[elem][param_name][0]
            # variable exceeds the maximum boundary
            if pos > bounds[elem][param_name][1]:
                vec_new[elem][param_name] = bounds[elem][param_name][1]
            # the variable is fine
            if bounds[elem][param_name][0] <= pos <= bounds[elem][param_name][1]:
                vec_new[elem][param_name] = pos
    return vec_new


def rand_1(pop, popsize, target_indx, mut, bounds):
    # mutation strategy
    # v = x_r1 + F * (x_r2 - x_r3)
    idxs = [idx for idx in range(popsize) if idx != target_indx]
    a, b, c = np.random.choice(idxs, 3, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]

    v_donor = {}
    for elem, param in x_1.items():
        v_donor[elem] = {}
        for param_name in param.keys():
            v_donor[elem][param_name] = x_1[elem][param_name] + mut * \
                                        (x_2[elem][param_name] - x_3[elem][param_name])
    v_donor = ensure_bounds(vec=v_donor, bounds=bounds)
    return v_donor


def best_1(pop, popsize, target_indx, mut, bounds, ind_sol):
    # mutation strategy
    # v = x_best + F * (x_r1 - x_r2)
    x_best = pop[ind_sol.index(np.max(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != target_indx]
    a, b = np.random.choice(idxs, 2, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]

    v_donor = {}
    for elem, param in x_best.items():
        v_donor[elem] = {}
        for param_name in param.items():
            v_donor[elem][param_name] = x_best[elem][param_name] + mut * \
                                        (x_1[elem][param_name] - x_2[elem][param_name])
    v_donor = ensure_bounds(vec=v_donor, bounds=bounds)
    return v_donor


def mutate(population, strategy, mut, bounds, ind_sol):
    mutated_indv = []
    for i in range(len(population)):
        if strategy == 'rand/1':
            v_donor = rand_1(pop=population, popsize=len(population), target_indx=i,
                             mut=mut, bounds=bounds)
        elif strategy == 'best/1':
            v_donor = best_1(pop=population, popsize=len(population), target_indx=i,
                             mut=mut, bounds=bounds, ind_sol=ind_sol)
        # elif strategy == 'current-to-best/1':
        #     v_donor = current_to_best_1(population, len(population), i, mut, bounds, ind_sol)
        # elif strategy == 'best/2':
        #     v_donor = best_2(population, len(population), i, mut, bounds, ind_sol)
        # elif strategy == 'rand/2':
        #     v_donor = rand_2(population, len(population), i, mut, bounds)
        mutated_indv.append(v_donor)
    return mutated_indv


def crossover(population, mutated_indv, crosspb):
    crossover_indv = []
    for i in range(len(population)):
        x_t = population[i]
        v_trial = {}
        for elem, param in x_t.items():
            v_trial[elem] = {}
            for param_name, pos in param.items():
                crossover_val = random.random()
                if crossover_val <= crosspb:
                    v_trial[elem][param_name] = mutated_indv[i][elem][param_name]
                else:
                    v_trial[elem][param_name] = x_t[elem][param_name]
        crossover_indv.append(v_trial)
    return crossover_indv


def create_selection_params(motors, population, cross_indv):
    if motors is not None and population is None:
        # hardware
        positions = [elm for elm in cross_indv]
        indv = {}
        for elem, param in motors.items():
            indv[elem] = {}
            for param_name, elem_obj in param.items():
                indv[elem][param_name] = (yield from bps.read(elem_obj))[elem]['value']
        positions.insert(0, indv)
        return positions
    if motors is None and population is not None:
        # sirepo simulation
        positions = [elm for elm in cross_indv]
        positions.insert(0, population[0])
        return positions


def create_rand_selection_params(motors, population, intensities, bounds):
    if motors is not None and population is None:
        # hardware
        positions = []
        change_indx = intensities.index(np.min(intensities))
        indv = {}
        for elem, param in motors.items():
            indv[elem] = {}
            for param_name, elem_obj in param.items():
                indv[elem][param_name] = (yield from bps.read(elem_obj))[elem]['value']
        positions.append(indv)
        indv = {}
        for elem, param in bounds.items():
            indv[elem] = {}
            for param_name, bound in param.items():
                indv[elem][param_name] = random.uniform(bound[0], bound[1])
        positions.append(indv)
        return positions, change_indx
    elif motors is None and population is not None:
        # sirepo simulation
        positions = []
        change_indx = intensities.index(np.min(intensities))
        positions.append(population[0])
        indv = {}
        for elem, param in bounds.items():
            indv[elem] = {}
            for param_name, bound in param.items():
                indv[elem][param_name] = random.uniform(bound[0], bound[1])
        positions.append(indv)
        return positions, change_indx


def select(population, intensities, motors, bounds, num_interm_vals,
           num_scans_at_once, uids, flyer_name, intensity_name, db):
    if motors is not None:
        # hardware
        new_population, new_intensities = omea_evaluation(motors=motors, bounds=bounds, popsize=None,
                                                          num_interm_vals=None, num_scans_at_once=None,
                                                          uids=uids, flyer_name=flyer_name,
                                                          intensity_name=intensity_name, db=db)
    else:
        new_population, new_intensities = omea_evaluation(motors=None, bounds=bounds, popsize=len(population) + 1,
                                                          num_interm_vals=num_interm_vals,
                                                          num_scans_at_once=num_scans_at_once,
                                                          uids=uids, flyer_name=flyer_name,
                                                          intensity_name=intensity_name, db=db)
    del new_population[0]
    del new_intensities[0]
    assert len(new_population) == len(population)
    for i in range(len(new_intensities)):
        if new_intensities[i] > intensities[i]:
            population[i] = new_population[i]
            intensities[i] = new_intensities[i]
    if motors is None:
        population.reverse()
        intensities.reverse()
    return population, intensities


def optimization_plan(fly_plan, bounds, db, motors=None, detector=None, max_velocity=0.2, min_velocity=0,
                      start_det=None, read_det=None, stop_det=None, watch_func=None,
                      run_parallel=None, num_interm_vals=None, num_scans_at_once=None, sim_id=None,
                      server_name=None, root_dir=None, watch_name=None, popsize=5, crosspb=.8, mut=.1,
                      mut_type='rand/1', threshold=0, max_iter=100, flyer_name='hardware_flyer',
                      intensity_name='intensity', opt_type='hardware'):
    """
    Optimize beamline using hardware flyers and differential evolution

    Custom plan to optimize motor positions of the TES beamline using differential evolution

    Parameters
    ----------
    fly_plan : callable
        Fly scan plan for current type of flyer.
        Currently the only option is `run_hardware_fly`, but another will be added for sirepo simulations
    bounds : dict of dicts
        Keys are motor names and values are dicts of low and high bounds. See format below.
        {'motor_name': {'low': lower_bound, 'high': upper_bound}}
    db : databroker.Broker
        databroker V1 instance
    motors : dict
        Keys are motor names and values are motor objects
    detector : detector object or None
        Detector to use, or None if no detector will be used
    max_velocity : float, optional
        Absolute maximum velocity for all motors
        Default is 0.2
    min_velocity : float, optional
        Absolute minimum velocity for all motors
    start_det : callable
        Function to start detector
    read_det : callable
        Function to read detector
    stop_det : callable
        Function to stop detector
    watch_func : callable
        Function to 'watch' positions/intensities/time as hardware
        moves or is read
    run_parallel : bool
        Run simulations in parallel
    num_interm_vals : int
        Number of positions to check in between individuals
    num_scans_at_once : int
        Number of scans to run at one time
    sim_id : str
        Simulation id
        Last 8 symbols of simulation URL
    server_name : str
        Name of server Sirepo runs on
    root_dir : str
        Path to store databroker documents
    watch_name : str
        Name of watch point to use as detector
    popsize : int, optional
        Size of population
    crosspb : float, optional
        Probability of crossover. Must be in range [0, 1]
    mut : float, optional
        Mutation factor. Must be in range [0, 1]
    mut_type : {'rand/1', 'best/1'}, optional
        Mutation strategy to use. 'rand/1' chooses random individuals to compare to.
        'best/1' uses the best individual to compare to.
        Default is 'rand/1'
    threshold : float, optional
        Threshold that intensity must be greater than or equal to to stop execution
    max_iter : int, optional
        Maximum iterations to allow
    flyer_name : str, optional
        Name of flyer. DataBroker stream name
        Default is 'hardware_flyer'
    intensity_name : {'intensity', 'mean'}, optional
        Use 'intensity' for hardware optimization or 'mean' for Sirepo optimization
        Default is 'intensity'
    opt_type : {'hardware', 'sirepo'}
        Type of optimization to perform
        Default is 'hardware'
    """
    needed_param_names = {'hardware': ['motors', 'start_det', 'read_det', 'stop_det', 'watch_func'],
                          'sirepo': ['run_parallel', 'num_interm_vals', 'num_scans_at_once', 'sim_id',
                                     'server_name', 'root_dir', 'watch_name']}
    if opt_type == 'hardware':
        # make sure all required parameters needed for hardware optimization aren't None
        needed_params = [motors, start_det, read_det, stop_det, watch_func]
        if any(p is None for p in needed_params):
            invalid_params = []
            for p in range(len(needed_params)):
                if needed_params[p] is None:
                    invalid_params.append(needed_param_names['hardware'][p])
            raise ValueError(f'The following parameters are set to None, but '
                             f'need to be set: {invalid_params}')
        # check if bounds passed in are within the actual bounds of the motors
        check_opt_bounds(motors, bounds)
        # create initial population
        initial_population = []
        for i in range(popsize):
            indv = {}
            if i == 0:
                for elem, param in motors.items():
                    indv[elem] = {}
                    for param_name, elem_obj in param.items():
                        indv[elem][param_name] = (yield from bps.read(elem_obj))[elem]['value']
            else:
                for elem, param in bounds.items():
                    indv[elem] = {}
                    for param_name, bound in param.items():
                        indv[elem][param_name] = random.uniform(bound[0], bound[1])
            initial_population.append(indv)
        uid_list = (yield from fly_plan(motors=motors, detector=detector, population=initial_population,
                                        max_velocity=max_velocity, min_velocity=min_velocity,
                                        start_det=start_det, read_det=read_det, stop_det=stop_det,
                                        watch_func=watch_func))
        pop_positions, pop_intensity = omea_evaluation(motors=motors, bounds=None, popsize=None,
                                                       num_interm_vals=None, num_scans_at_once=None,
                                                       uids=uid_list, flyer_name=flyer_name,
                                                       intensity_name=intensity_name, db=db)
    elif opt_type == 'sirepo':
        # make sure all required parameters needed for sirepo optimization aren't None
        needed_params = [run_parallel, num_interm_vals, num_scans_at_once, sim_id, server_name,
                         root_dir, watch_name]
        if any(p is None for p in needed_params):
            invalid_params = []
            for p in range(len(needed_params)):
                if needed_params[p] is None:
                    invalid_params.append(needed_param_names['sirepo'][p])
            raise ValueError(f'The following parameters are set to None, but '
                             f'need to be set: {invalid_params}')
        # Initial population
        initial_population = []
        for i in range(popsize):
            indv = {}
            for elem, param in bounds.items():
                indv[elem] = {}
                for param_name, bound in param.items():
                    indv[elem][param_name] = random.uniform(bound[0], bound[1])
            initial_population.append(indv)
        first_optic = list(bounds.keys())[0]
        first_param_name = list(bounds[first_optic].keys())[0]
        initial_population = sorted(initial_population, key=lambda kv: kv[first_optic][first_param_name])
        uid_list = (yield from fly_plan(population=initial_population, num_interm_vals=num_interm_vals,
                                        num_scans_at_once=num_scans_at_once, sim_id=sim_id,
                                        server_name=server_name, root_dir=root_dir, watch_name=watch_name,
                                        run_parallel=run_parallel))
        pop_positions, pop_intensity = omea_evaluation(motors=None, bounds=bounds, popsize=len(initial_population),
                                                       num_interm_vals=num_interm_vals,
                                                       num_scans_at_once=num_scans_at_once, uids=uid_list,
                                                       flyer_name=flyer_name, intensity_name=intensity_name, db=db)
        pop_positions.reverse()
        pop_intensity.reverse()
    else:
        raise ValueError(f'Opt_type {opt_type} is invalid. Choose either hardware or sirepo')
    # Termination conditions
    v = 0  # generation number
    consec_best_ctr = 0  # counting successive generations with no change to best value
    old_best_fit_val = 0
    best_fitness = [0]
    all_uids = {}
    while not ((v > max_iter) or (consec_best_ctr >= 5 and old_best_fit_val >= threshold)):
        print(f'GENERATION {v + 1}')
        best_gen_sol = []
        # mutate
        mutated_trial_pop = mutate(population=pop_positions, strategy=mut_type, mut=mut,
                                   bounds=bounds, ind_sol=pop_intensity)
        # crossover
        cross_trial_pop = crossover(population=pop_positions, mutated_indv=mutated_trial_pop,
                                    crosspb=crosspb)
        # select
        if opt_type == 'hardware':
            select_positions = yield from create_selection_params(motors=motors, population=None,
                                                                  cross_indv=cross_trial_pop)
            uid_list = (yield from fly_plan(motors=motors, detector=detector, population=select_positions,
                                            max_velocity=max_velocity, min_velocity=min_velocity,
                                            start_det=start_det, read_det=read_det, stop_det=stop_det,
                                            watch_func=watch_func))
            pop_positions, pop_intensity = select(population=pop_positions, intensities=pop_intensity,
                                                  motors=motors, bounds=None, num_interm_vals=None,
                                                  num_scans_at_once=None, uids=uid_list,
                                                  flyer_name=flyer_name, intensity_name=intensity_name, db=db)
        else:
            select_positions = yield from create_selection_params(motors=None, population=pop_positions,
                                                                  cross_indv=cross_trial_pop)
            uid_list = (yield from fly_plan(population=select_positions, num_interm_vals=num_interm_vals,
                                            num_scans_at_once=num_scans_at_once, sim_id=sim_id,
                                            server_name=server_name, root_dir=root_dir,
                                            watch_name=watch_name, run_parallel=run_parallel))
            pop_positions, pop_intensity = select(population=pop_positions, intensities=pop_intensity,
                                                  motors=None, bounds=bounds, num_interm_vals=num_interm_vals,
                                                  num_scans_at_once=num_scans_at_once, uids=uid_list,
                                                  flyer_name=flyer_name, intensity_name=intensity_name, db=db)
        all_uids[f'gen-{v + 1}'] = uid_list

        # get best solution
        gen_best = np.max(pop_intensity)
        best_indv = pop_positions[pop_intensity.index(gen_best)]
        best_gen_sol.append(best_indv)
        best_fitness.append(gen_best)

        print('      > FITNESS:', gen_best)
        print('         > BEST POSITIONS:', best_indv)

        v += 1
        if np.round(gen_best, 6) == np.round(old_best_fit_val, 6):
            consec_best_ctr += 1
            print('Counter:', consec_best_ctr)
        else:
            consec_best_ctr = 0
        old_best_fit_val = gen_best

        if consec_best_ctr >= 5 and old_best_fit_val >= threshold:
            print('Finished')
            break
        else:
            if opt_type == 'hardware':
                positions, change_indx = yield from create_rand_selection_params(motors=motors, population=None,
                                                                                 intensities=pop_intensity,
                                                                                 bounds=bounds)
                uid_list = (yield from fly_plan(motors=motors, detector=detector, population=positions,
                                                max_velocity=max_velocity, min_velocity=min_velocity,
                                                start_det=start_det, read_det=read_det, stop_det=stop_det,
                                                watch_func=watch_func))
                rand_pop, rand_int = select(population=[pop_positions[change_indx]],
                                            intensities=[pop_intensity[change_indx]],
                                            motors=motors, bounds=bounds, num_interm_vals=None,
                                            num_scans_at_once=None, uids=uid_list, flyer_name=flyer_name,
                                            intensity_name=intensity_name, db=db)
            else:
                positions, change_indx = yield from create_rand_selection_params(motors=None,
                                                                                 population=pop_positions,
                                                                                 intensities=pop_intensity,
                                                                                 bounds=bounds)
                uid_list = (yield from fly_plan(population=positions, num_interm_vals=num_interm_vals,
                                                num_scans_at_once=num_scans_at_once, sim_id=sim_id,
                                                server_name=server_name, root_dir=root_dir, watch_name=watch_name,
                                                run_parallel=run_parallel))
                rand_pop, rand_int = select(population=[pop_positions[change_indx]],
                                            intensities=[pop_intensity[change_indx]], motors=None, bounds=bounds,
                                            num_interm_vals=num_interm_vals, num_scans_at_once=num_scans_at_once,
                                            uids=uid_list, flyer_name=flyer_name, intensity_name=intensity_name,
                                            db=db)
            all_uids[f'gen-{v}'] += uid_list

            assert len(rand_pop) == 1 and len(rand_int) == 1
            pop_positions[change_indx] = rand_pop[0]
            pop_intensity[change_indx] = rand_int[0]

    # best solution overall should be last one
    optimized_positions = best_gen_sol[-1]
    print('\nThe best individual is', optimized_positions, 'with a fitness of', gen_best)
    print('It took', v, 'generations')

    if opt_type == 'hardware':
        print('Moving to optimal positions')
        yield from move_to_optimized_positions(motors, optimized_positions)
        print('Done')

    print(f"Convergence list: {best_fitness}")

    yield from bp.count([], md={'best_fitness': best_fitness, 'optimized_positions':
                                optimized_positions, 'uids': all_uids})
