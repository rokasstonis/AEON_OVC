from aeon.dj_pipeline import acquisition, streams
exp = {"experiment_name": "ProjectAeonOVC-2026-09-14_001"}
print(len(acquisition.Chunk & exp), len(streams.CameraVideo & exp & {"device_name": "CameraTop"}))
print(len(streams.SoundCardSoundEvents & exp), len(acquisition.LightEvents & exp), len(acquisition.MessageLog & exp))