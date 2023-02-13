

gpo = gp.Optimizer(init_scheme='quasi-random', n_init=4, run_engine=RE, db=db, shutter=psh, detector=vstream, dofs=dofs, dof_bounds=hard_bounds, init_training_iter=200, verbose=True)

learn_timeout = 600
learn_start = ttime.monotonic()
while ttime.monotonic() - learn_start < learn_timeout:

    gpo.learn(n_iter=1, n_per_flight=4, strategy='maximize_expected_information', reuse_hypers=False)
    gpo.learn(n_iter=1, n_per_flight=4, strategy='maximize_expected_improvement', reuse_hypers=False)

timestamps = gpo.data.time.astype(int).values / 1e9

plt.plot(timestamps[1:] - timestamps[0], [np.nanmax(gpo.fitness[:i]) for i in range(1, len(gpo.fitness))])

gpo.data['fitness'] = gpo.fitness

gpo.data.drop(columns=[f'{gpo.detector.name}_image'], inplace=True)

gpo.data.to_hdf(f'data/{int(timestamps[0])}.h5', 'data')

del gpo
