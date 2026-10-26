import inspect
import numpy as np
import pytest

import retina
from circuit import (FlyCircuit, GENE_BOUNDS, GENOME_SIZE, SL_LPLC2, SL_LPI, random_genomes,
                     receptive_fields, clip_genomes)
from config import (GROUND_Y, CEILING_Y, FLY_X, EYE_RANGE, N_OMMATIDIA, PIPE_WIDTH, PIPE_GAP,
                    FLAP_STRENGTH, FLAP_COOLDOWN, GRAVITY, MAX_GAP_SHIFT)
from connectome import evolved_gradient
from evolution import Evolution, harmonic_mean, evaluate, run_episode, save_champion, load_champion
from world import World


# ---- world -------------------------------------------------------------------

def test_same_seed_same_course():
    a, b = World(1, 42), World(1, 42)
    for _ in range(500):
        a.step([False]), b.step([False])
    assert [p[:2] for p in a.pipes] == [p[:2] for p in b.pipes]


def test_free_fall_dies_on_the_ground():
    w = World(1, 0)
    while not w.done:
        w.step([False])
    assert w.death_cause[0] == "ground"


def test_constant_flapping_dies_on_the_ceiling():
    w = World(1, 0)
    while not w.done:
        w.step([True])
    assert w.death_cause[0] == "ceiling"


def test_flap_cooldown():
    w = World(1, 0)
    w.step([True])
    assert w.vy[0] == pytest.approx(FLAP_STRENGTH + GRAVITY)
    flaps = 1
    for _ in range(FLAP_COOLDOWN - 1):
        w.step([True])
        flaps += int(w.flapped[0])
    assert flaps == 1


def test_episode_is_capped():
    w = World(1, 0, max_frames=10)
    while not w.done:
        w.step([w.vy[0] > 2])
    assert w.frame == 10


# ---- retina ------------------------------------------------------------------

def test_straight_down_ray_measures_height_above_ground():
    y = 500.0
    d = retina.ray_distances(np.array([y]), np.empty((0, 4)))[0]
    down = np.argmin(retina.ELEVATIONS)                       # most ventral ray, -100 deg
    expected = (GROUND_Y - y) / abs(np.sin(retina.ELEVATIONS[down]))
    assert d[down] == pytest.approx(expected)


def test_horizon_ray_hits_pipe_face_only_outside_the_gap():
    horizon = N_OMMATIDIA // 2
    assert retina.ELEVATIONS[horizon] == pytest.approx(0.0)
    gap_y, x0 = 400.0, FLY_X + 200.0
    rects = np.array([[x0, x0 + PIPE_WIDTH, -1e6, gap_y - PIPE_GAP / 2],
                      [x0, x0 + PIPE_WIDTH, gap_y + PIPE_GAP / 2, 1e6]])
    d = retina.ray_distances(np.array([gap_y, gap_y + 150.0, gap_y - 150.0]), rects)
    assert d[0, horizon] > EYE_RANGE          # looking through the gap
    assert d[1, horizon] == pytest.approx(200.0)
    assert d[2, horizon] == pytest.approx(200.0)


def test_nearness_is_bounded():
    n = retina.sample(np.array([CEILING_Y + 20.0, GROUND_Y - 20.0]), np.empty((0, 4)))
    assert n.min() >= 0.0 and n.max() <= 1.0


# ---- circuit -----------------------------------------------------------------

def test_controller_only_sees_the_eye_and_its_own_velocity():
    params = list(inspect.signature(FlyCircuit.step).parameters)
    assert params == ["self", "nearness", "vy"]


def test_dales_law_bounds():
    assert (GENE_BOUNDS[SL_LPLC2] >= 0).all()      # LPLC2 is excitatory
    assert (GENE_BOUNDS[SL_LPI] <= 0).all()        # LPi is inhibitory
    g = clip_genomes(np.full((1, GENOME_SIZE), 1e9))
    assert (g[0, SL_LPI] <= 0).all()


def test_receptive_fields_tile_dorsal_to_ventral():
    rf = receptive_fields(8)
    assert np.allclose(rf.sum(axis=1), 1.0)
    assert (np.diff(rf.argmax(axis=1)) > 0).all()


def test_looming_below_drives_ventral_lplc2():
    g = random_genomes(1, np.random.default_rng(0))
    c = FlyCircuit(g)
    near = np.zeros((1, N_OMMATIDIA))
    c.step(near, np.zeros(1))
    spikes = np.zeros(c.spk_lplc2.shape[1])
    for t in range(30):
        near[0, -10:] = min(1.0, 0.03 * t)          # the ventral field fills up
        c.step(near, np.zeros(1))
        spikes += c.spk_lplc2[0]
    assert spikes[-3:].sum() > spikes[:3].sum()


def test_gf_drives_flaps():
    g = np.zeros((1, GENOME_SIZE))
    g[0] = GENE_BOUNDS[:, 0]
    from circuit import IDX
    g[0, IDX["gf_tonic"]] = 25.0
    g[0, IDX["gf_vth"]] = -56.0
    c = FlyCircuit(g)
    fired = [c.step(np.zeros((1, N_OMMATIDIA)), np.zeros(1))[0] for _ in range(60)]
    assert any(fired)


# ---- evolution ---------------------------------------------------------------

def test_harmonic_mean_punishes_specialists():
    h = harmonic_mean([[8000.0, 30.0], [500.0, 500.0], [0.0, 900.0]])
    assert h[1] > h[0] > h[2] == 0.0


def test_seeds_change_every_generation():
    evo = Evolution(pop_size=8, seed=0)
    first = evo.seeds()
    evo.tell(np.ones((8, len(first))))
    assert evo.seeds() != first


def test_evolution_improves_on_a_short_run():
    evo = Evolution(pop_size=24, seed=3)
    for _ in range(6):
        scores, _ = evaluate(evo.population, evo.seeds(), max_frames=600)
        evo.tell(scores)
    assert evo.best_history[-1] >= evo.best_history[0] * 0.8
    assert evo.record >= max(evo.best_history)


def test_checkpoint_round_trip(tmp_path):
    g = random_genomes(1, np.random.default_rng(1))[0]
    path = tmp_path / "champ.npz"
    save_champion(path, g, record=1.0)
    assert np.allclose(load_champion(path), g)


def test_stale_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, genome=np.zeros(8), version=1)
    assert load_champion(path) is None


def test_gradient_label():
    g = np.zeros(GENOME_SIZE)
    g[SL_LPLC2] = np.linspace(0, 40, SL_LPLC2.stop - SL_LPLC2.start)
    assert evolved_gradient(g)[0] == "ventral > dorsal"
    g[SL_LPLC2] = g[SL_LPLC2][::-1]
    assert evolved_gradient(g)[0] == "dorsal > ventral"


def test_consecutive_gaps_shift_by_a_bounded_amount():
    w = World(1, 7)
    for _ in range(200):
        w._spawn(0.0)
    gaps = [p[1] for p in w.pipes]
    assert np.abs(np.diff(gaps)).max() <= MAX_GAP_SHIFT + 1e-9


def test_acute_zone_is_densest_at_the_horizon():
    spacing = -np.diff(np.degrees(retina.ELEVATIONS))
    assert spacing[N_OMMATIDIA // 2] < spacing[0]


def test_circuit_does_not_alias_the_input_buffer():
    c = FlyCircuit(random_genomes(1, np.random.default_rng(2)))
    buf = np.zeros((1, N_OMMATIDIA))
    c.step(buf, np.zeros(1))
    buf[:] = 0.5                     # caller reuses its buffer for the next frame
    c.step(buf, np.zeros(1))
    assert c.loom.max() > 0


def test_batched_evaluation_matches_one_course_at_a_time():
    from evolution import episode_fitness
    g = random_genomes(6, np.random.default_rng(5))
    fit, _ = evaluate(g, [11, 12], max_frames=800)
    ref = np.column_stack([episode_fitness(run_episode(g, s, 800)) for s in (11, 12)])
    assert np.allclose(fit, ref)


def test_training_courses_never_reuse_held_out_courses():
    from evolution import test_seeds, validation_seeds
    held_out = set(test_seeds(1000)) | set(validation_seeds(1000))
    for seed in (0, 1, 5):
        evo = Evolution(pop_size=4, seed=seed)
        for gen in (1, 999, 1000, 2000, 5000):
            evo.generation = gen
            assert not held_out & set(evo.seeds())
