from foragingabc.dj_pipeline import pose
r = {"experiment_name": "ProjectAeonOVC-2026-09-14_001", "sleap_method": "ovc_top_260921_legacy"}
(pose.SleapRawTracking & r).delete()   # answer yes at the prompt
(pose.PoseQC.jobs & r).delete()
