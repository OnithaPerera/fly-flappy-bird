import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
pygame = pytest.importorskip("pygame")


@pytest.fixture(scope="module")
def app():
    pygame.init()
    pygame.display.set_mode((16, 16))
    import main
    if main.load_champion(main.CHECKPOINT) is None:
        pytest.skip("no champion_genome.npz")
    return main.App(SimpleNamespace(seed=0, replay=True))


def test_replay_records_each_course_once(app):
    while not app.finished:
        app.step()
    for _ in range(50):              # stepping a finished course must not re-score it
        app.step()
    assert len(app.replay_results) == 1
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
    assert not app.finished and app.course == 1


def test_hud_has_every_field_the_dashboard_reads(app):
    hud = app.hud()
    for field in ("mode", "course", "pipes", "near", "loom", "currents", "genome", "trace", "replay_results"):
        assert hasattr(hud, field)
