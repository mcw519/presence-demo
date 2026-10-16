// Settings: schema (defaults, ranges, help text), modules in data-flow order, presets, settings panel.
// Classic script; all files share one global scope and load in order from index.html.
// ---------- settings schema ----------
const DEF={face_roi:[0.2,0.05,0.8,0.95],mirror_view:true,show_face_roi:true,pose_enabled:false,pose_conf:0.4,pose_roi:[0,0,1,1],show_pose_roi:true,roi_rule:'either',
 turned_away_early:true,turned_away_seconds:2,recede_ratio:0.1,distance_enabled:false,camera_hfov:60,face_width_m:0.15,shoulder_width_m:0.38,
 distance_calib:1,distance_ema:0.3,zone_min:0.4,zone_max:2,leave_distance:2.5,min_conf:0.4,embed_backend:'fp32',npu_precision:'int8',cos_match_int8:0.4,cos_match:0.4,cos_weak:0.2,iou_match:0.3,embed_ema:0.15,join_seconds:2,grace_seconds:15,
 present_frames:2,absent_frames:3,idle_after:3,present_timeout:15,empty_timeout:5,base_timeout:20,maybe_away_frac:0.5,end_after_away:5,new_group_seconds:3,
 priority_mode:'vision',speech_join:'auto',presence_model:'rules',intent_model:'rules',intent_tau:1,join_threshold:0.6,drop_threshold:0.35,drop_seconds:8,intent_weighted_presence:false,w_facing:0.45,w_distance:0.1,w_dwell:0.1,w_approach:0.05,w_still:0.15,w_center:0.15,leave_halflife:1.5,detect_gain:4,lowconf_factor:0.15,p_active:0.8,p_maybe:0.6,p_away:0.3,away_hold_seconds:1,tracker_mode:'legacy',coast_seconds:2,coast_present_conf:0.4,gate_mahalanobis:9.21,appearance_min_sim:0.6,appearance_ema:0.2,body_min_conf:0.4,present_max_silence:0,motion_gate:true,motion_threshold:0.015,force_detect_seconds:2,embed_throttle:true,embed_interval:3,embed_move_thresh:0.15,adaptive_fps:true,fps_low:0.5,fps_high:2,roi_crop:true,crop_margin:0.1,max_width:640,fps:1,vad_k:2.5,vad_min:0.005,vad_hang:400,
 audio_pipeline:'energy_vad',iso_onset_guard:false,iso_reset_on_change:true,onset_needs_vision:true,onset_entry_only:true,clean_vad_margin_db:10,clean_vad_min_db:-50,clean_vad_hang:0.3,xcheck_enabled:true,near_range:1,xcheck_hold:0.75,xcheck_stale:2,xcheck_require_calib:false,xcheck_floor_db:-40,xcheck_recover_mix:0.3,xcheck_attack:0.01,xcheck_release:0.05,asr_provider:'none',asr_source:'clean',asr_feed:'speech',asr_region:'global',asr_speech_model:'universal-3-6-pro',asr_language_codes:'zh,en',asr_keyterms:'',asr_format_turns:true,asr_eot_confidence:0.4,asr_min_turn_silence:400,asr_max_turn_silence:1280,reid_enabled:false,reid_consent:false,reid_show_crops:true,reid_provider:'jev',reid_model:'jev-latest',reid_threshold:0.9,reid_uncertain_low:0.4,reid_ttl:600,reid_max_refs:3,reid_min_face:0.06,reid_decision_timeout:10,reid_face_wait:3,reid_timeout:2,reid_cooldown:5,reid_cap_per_hour:30,reid_cap_total:200,reid_cost_per_call:0,monitor:'off'};
const CLIENT=['monitor','mirror_view','fps','vad_k','vad_min','vad_hang','show_face_roi','show_pose_roi'];
const BOXES=['face_roi','pose_roi'];
const S=(l,h,min,max,step,u='')=>({l,h,min,max,step,u});
const GROUPS=[
 ['判斷優先',{priority_mode:{l:'判斷優先',h:'影像優先：畫面有人就不會因為靜默離開，ROI 確定沒人就照 ROI 空的逾時離開（不管聲音），聲音只在影像不確定時使用，而且單靠聲音不會從 AWAY 恢復，適合吵雜環境。平衡：聲音和影像一起決定。聲音優先：主要看靜默，影像只微調逾時。',
   opts:{vision:'影像優先',balanced:'平衡',audio:'聲音優先'}},
   speech_join:{l:'說話立即加入',h:'有人說話時，是否不用等「加入時間」就讓 ROI 內的人成為參與者（聲音沒有位置資訊）。自動：影像優先時關閉，其他模式用「唯一候選人」。唯一候選人：只有一位可見的候選人、臉在臉部 ROI 內、正面朝向（開距離時還要在對話區內）才立即加入。全部：所有可見的候選人都立即加入。',
   opts:{auto:'自動',off:'關閉',single_candidate:'唯一候選人',all:'全部'}},
   present_max_silence:S('有人時最長靜默（影像優先）','影像優先模式下，人在畫面裡但一直不說話，最多等多久就判定離開。0＝不限制。',0,3600,10,'s')}],
 ['偵測範圍',{face_roi:{...S('臉部 ROI (x0,y0,x1,y1)','臉的中心點要落在這個框（青色）裡才算數。可以直接在影像上拖曳或拉角調整。',0,1,0.01),box:true},
   mirror_view:{l:'鏡像顯示',h:'像照鏡子一樣左右翻轉預覽畫面與標示框（自拍視角），比較直覺。只影響這個瀏覽器的顯示：送到伺服器的影像不翻轉，和機器人相機看到的一樣；ROI 數值也一律用原始相機座標儲存（鏡像時畫面上的左邊＝x 值大的一側）。',bool:true},
   show_face_roi:{l:'顯示臉部 ROI',h:'在影像上顯示青色的臉部 ROI 框（只影響顯示）。',bool:true},
   min_conf:S('最低偵測信心','低於這個值的偵測不算有人也不算沒人，會退回只看聲音。',0,1,0.01)}],
 ['身分比對',{cos_match:S('同一人相似度門檻','臉部 embedding 的 cosine 相似度達到這個值就算同一人（SFace 約 0.36）。',0,1,0.01),
   cos_weak:S('弱相似度（位置重疊時）','框的位置大致重疊時，相似度只要達到這個較低的值也算同一人。',0,1,0.01),
   embed_backend:{l:'嵌入後端',h:'計算人臉特徵（SFace）的方式。fp32：原始模型，最準（預設）。int8：opencv_zoo 官方的 INT8 量化版，檔案小 4 倍；在這台 x86 伺服器上反而比較慢，但在 ARM 手機上通常比較快。npu：轉成 TFLite，在 Android 上可以交給 NPU（NNAPI／NeuroPilot）計算；伺服器上沒有 NPU，會用 CPU 執行，並在「資源監控」顯示實際使用的加速器。',opts:{fp32:'fp32',int8:'int8',npu:'npu (TFLite)'}},
   npu_precision:{l:'NPU 模型精度',h:'npu 後端使用的 TFLite 模型。int8：全整數量化，最適合 NPU。fp16：半精度，適合 GPU；也是沒有 NPU 時 CPU 最快的選項（int8 在沒有 NPU 時會自動改用 fp16）。',opts:{int8:'int8',fp16:'fp16'}},
   cos_match_int8:S('同一人門檻（量化模型）','使用 int8 或 npu int8 後端時的「同一人」相似度門檻。實測量化模型和 fp32 的判定一致率約 99.9%，預設和 fp32 相同。',0,1,0.01),
   iou_match:S('位置重疊門檻 IoU','前後兩張畫面的框重疊多少算「位置一樣」。',0,1,0.01),
   embed_ema:S('模板更新速度','每張畫面更新多少比例的臉部模板。越大越能適應光線變化，但也越容易被別人帶偏。',0,1,0.01)}],
 ['加入與防閃爍',{join_seconds:S('加入參與者所需時間','在 ROI 裡待滿這麼久才算參與者（開口說話會立即加入）。',0,30,0.5,'s'),
   grace_seconds:S('消失保留時間','參與者看不到超過這麼久才移除，避免短暫被擋住就被踢掉。',0,120,1,'s'),
   present_frames:S('確認有人所需張數','連續幾張畫面有人才算「有人」。',1,20,1,'張'),
   absent_frames:S('確認沒人所需張數','連續幾張畫面沒人才算「ROI 空」，可以防止偵測閃爍。',1,20,1,'張')}],
 ['逾時',{idle_after:S('靜默轉 IDLE','靜默多久會從 ACTIVE 轉成 IDLE。',0,60,0.5,'s'),
   present_timeout:S('有人時的逾時','平衡模式：畫面裡有參與者但都不說話時，靜默多久判定離開。預設 15 s，比影像不確定時的 20 s 短。',1,600,1,'s'),
   empty_timeout:S('ROI 空時的逾時','沒人又靜默時，多快判定離開（縮短）。',1,120,0.5,'s'),
   base_timeout:S('影像不確定時的逾時','影像信心太低時，只看聲音的逾時。',1,300,1,'s'),
   maybe_away_frac:S('MAYBE_AWAY 比例','逾時進行到這個比例時進入 MAYBE_AWAY。',0.05,0.95,0.05),
   end_after_away:S('AWAY 轉 ENDED','AWAY 持續多久後結束對話並清除 embedding。',0,120,1,'s'),
   new_group_seconds:S('換組確認時間','舊參與者都不見、新的一群人出現多久後觸發 NEW_GROUP。',0,60,0.5,'s')}]];
const B=(l,h)=>({l,h,bool:true});
GROUPS.push(['互動意圖',{intent_model:{l:'加入規則',h:'規則：在 ROI 待滿「加入時間」就成為參與者（目前預設）。意圖分數：每個人有 0–1 的互動意圖（是否面向機器人、距離、停留、是否走近、是否只是路過、是否在畫面中央、說話），分數高於門檻並持續「加入時間」才加入；路人、旁觀者不會加入，也不會讓對話一直持續。',
   opts:{rules:'規則',score:'意圖分數'}},
   join_threshold:S('加入門檻','意圖分數模式：意圖高於這個值並持續「加入時間」，才成為參與者（也才會開始新的對話）。',0.05,0.99,0.01),
   drop_threshold:S('降為旁觀者門檻','意圖分數模式：參與者的意圖低於這個值並持續「降級時間」，變成旁觀者（最後一位參與者不會被降級，離開與否由在場機率決定）。',0,0.95,0.01),
   drop_seconds:S('降級時間','意圖低於門檻要持續多久才降為旁觀者。',0,120,1,'s'),
   intent_tau:S('平滑時間','意圖分數的平滑時間常數，越大越穩定但反應越慢。',0.1,20,0.1,'s'),
   intent_weighted_presence:B('在場機率依意圖加權','開啟後（機率模式），參與者的在場機率乘上意圖比例：意圖低的參與者比較撐不住對話。'),
   w_facing:S('權重：面向','臉是否正對（YuNet 五點）＋肩膀方向（姿勢）。面向也會整體縮放分數：背對的人很難得到高分。',0,1,0.05),
   w_distance:S('權重：距離','在對話距離內＝1，到「離開距離」降為 0。距離估計關閉時自動忽略。',0,1,0.05),
   w_dwell:S('權重：停留','在 ROI 待越久越高（4 s 滿分）。',0,1,0.05),
   w_approach:S('權重：走近','變大（走近）加分、變小（走遠）扣分。',0,1,0.05),
   w_still:S('權重：沒在路過','橫向移動越快分數越低，用來排除走過去的路人。',0,1,0.05),
   w_center:S('權重：位置','越接近 ROI 中央越高。',0,1,0.05)}]);
GROUPS.push(['在場機率',{presence_model:{l:'在場判斷模型',h:'規則：用連續幀計數（有人／沒人）加逾時判斷（目前預設）。機率：每個人有一個「在場機率」，看到就往上、沒看到就往下，最後用整群的機率加門檻決定狀態，比較不怕偵測閃爍和短暫遮擋。搭配 Kalman 追蹤效果最好。',
   opts:{rules:'規則',hmm:'機率'}},
   leave_halflife:S('離開半衰期','機率模式：完全沒看到人時，「在場」的勝算每隔這麼多秒減半。越小越快判定離開。預設 0.6 s 約等於 ROI 空 5 秒判定 AWAY。',0.1,30,0.1,'s'),
   detect_gain:S('偵測證據強度','機率模式：每秒偵測到人時，機率往上升的速度（乘上偵測信心）。越大回到 ACTIVE 越快。',0.1,20,0.1),
   lowconf_factor:S('低信心衰減比例','機率模式：只有低信心偵測時，機率下降速度是「完全沒看到」的多少倍。小一點可以在光線不好時不輕易判定離開。',0,1,0.05),
   p_active:S('ACTIVE 門檻','機率模式：群組機率高於這個值，才從 MAYBE_AWAY／AWAY 回到 ACTIVE。',0.05,0.99,0.01),
   p_maybe:S('MAYBE_AWAY 門檻','機率模式：群組機率低於這個值，進入 MAYBE_AWAY。',0.02,0.98,0.01),
   p_away:S('AWAY 門檻','機率模式：群組機率低於這個值並持續「AWAY 確認時間」，判定 AWAY。',0.01,0.95,0.01),
   away_hold_seconds:S('AWAY 確認時間','機率模式：機率要持續低於 AWAY 門檻多久才判定 AWAY。',0,30,0.5,'s')}]);
GROUPS.push(['追蹤',{tracker_mode:{l:'追蹤模式',h:'傳統：IoU + 臉部相似度，逐一貪婪配對（目前預設）。Kalman：每個人用 Kalman 濾波預測位置和速度，結合 IoU、移動距離、臉部相似度和衣服顏色（軀幹 HSV 直方圖，不需要模型），用匈牙利演算法一次配對；短暫被擋住時沿著預測位置繼續追蹤（虛線框）。',
   opts:{legacy:'傳統',kalman:'Kalman'}},
   coast_seconds:S('預測延續時間','Kalman：沒偵測到時，沿著預測位置繼續追蹤多久。之後就當作不可見（寬限期照常計算）。',0,10,0.5,'s'),
   coast_present_conf:S('預測算有人的門檻','Kalman：預測框的可見信心（從 1 隨時間降到 0）高於這個值時，仍然算「有人」。調高會更快判定離開。',0,1,0.05),
   gate_mahalanobis:S('移動距離門檻','Kalman：偵測位置和預測位置的統計距離（Mahalanobis²）超過這個值就不配對，除非臉部相似度很高。9.21 約等於 99%。',1,100,0.5),
   appearance_min_sim:S('衣服相似度門檻','Kalman：看不到臉時，衣服顏色相似度低於這個值就視為不同的人。',0,1,0.05),
   appearance_ema:S('衣服模板更新率','Kalman：每次配對後，衣服顏色模板用多少比例更新成新的樣子。',0,1,0.05)}]);
GROUPS.push(['姿勢',{pose_enabled:B('啟用姿勢偵測','用 MoveNet MultiPose Lightning（Apache-2.0，最多 6 人）偵測身體與關鍵點（肩、髖、鼻、眼）。側身或背對時沒有臉也能判斷有人。CPU 成本較高，預設關閉。'),
   pose_conf:S('身體偵測信心','低於這個值的身體偵測會被忽略。',0.05,1,0.05),
   body_min_conf:S('只有身體時的信心門檻','看不到臉、只偵測到身體時，信心達到這個值才算「有人」（有臉時用「偵測信心」）。',0.05,1,0.05),
   pose_roi:{...S('身體 ROI (x0,y0,x1,y1)','身體框的中心點落在這個框（紫色）裡才算數，通常比臉部 ROI 大。',0,1,0.01),box:true},
   show_pose_roi:{l:'顯示身體 ROI',h:'在影像上顯示紫色的身體 ROI 框（只影響顯示）。',bool:true},
   roi_rule:{l:'參與者判定規則',h:'臉或身體：臉在臉部 ROI，或身體在身體 ROI，任一成立就算。只看臉：只用臉部 ROI。只看身體：只用身體 ROI（需要開啟姿勢偵測）。',opts:{either:'臉或身體',face:'只看臉',body:'只看身體'}},
   turned_away_early:B('轉身離開提早預警','所有參與者都轉身（側面或背對），而且身形變小（往遠處走）時，提早進入 MAYBE_AWAY。'),
   turned_away_seconds:S('轉身持續時間','轉身＋遠離的狀態要持續多久才觸發。',0,30,0.5,'s'),
   recede_ratio:S('遠離判定比例','約 2 秒內身形（或臉寬）縮小超過這個比例，就視為正在遠離。',0.01,0.9,0.01)}]);
GROUPS.push(['距離',{distance_enabled:B('啟用距離估計','用臉寬（約 0.15 m）和肩寬（約 0.38 m，只在正面時使用）加上鏡頭視角估計距離。'),
   camera_hfov:S('鏡頭水平視角','鏡頭的水平 FOV。不知道的話用預設值，再用下方的校正。',20,170,1,'°'),
   face_width_m:S('平均臉寬','用於估計距離的平均臉寬。',0.08,0.3,0.005,'m'),
   shoulder_width_m:S('平均肩寬','用於估計距離的平均肩寬。',0.2,0.6,0.01,'m'),
   distance_calib:S('校正係數','估計距離乘上這個係數。可以用下方「我站在 X 公尺」自動計算。',0.2,5,0.01),
   distance_ema:S('平滑係數','每次新估計值的權重（EMA）。越小越平滑，但反應越慢。',0.01,1,0.01),
   zone_min:S('對話區最近','比這更近的標示為「太近」。',0,10,0.1,'m'),
   zone_max:S('對話區最遠','對話區的外緣。超過但還沒到離開距離時標示為「偏遠」。',0.1,20,0.1,'m'),
   leave_distance:S('離開距離','超過這個距離，就算還在畫面裡也視為已離開。',0.2,30,0.1,'m')}]);
GROUPS.push(['效能',{motion_gate:B('移動偵測閘門','畫面幾乎沒變化、狀態也穩定時，跳過人臉偵測，沿用上一次的結果。'),
   motion_threshold:S('變化門檻','縮小灰階畫面的平均差異（0–1）低於這個值就視為沒變化。有雜訊的鏡頭要調高。',0,0.2,0.001),
   force_detect_seconds:S('強制偵測間隔','即使畫面沒變化，最久每隔幾秒也要完整偵測一次。',0.5,30,0.5,'s'),
   embed_throttle:B('Embedding 節流','人臉沒移動時沿用上次的 embedding；新的人臉、移動或縮放超過門檻、相似度偏低，或超過間隔時間時才重新計算。'),
   embed_interval:S('Embedding 重算間隔','每張臉最久每隔幾秒重算一次 embedding。',0,30,0.5,'s'),
   embed_move_thresh:S('移動門檻','中心位移或大小變化超過臉框的這個比例就重算。',0,1,0.01),
   adaptive_fps:B('自適應 fps','由伺服器依狀態建議送幀速率：穩定時用低 fps，有變化或 MAYBE_AWAY 時用高 fps。開啟時會取代下方的「每秒送出張數」。'),
   fps_low:S('穩定時 fps','狀態穩定（ACTIVE/IDLE/ENDED、沒有新人臉）時的建議 fps。',0.2,5,0.1,'fps'),
   fps_high:S('變化時 fps','有新人臉、參與者消失、MAYBE_AWAY 時的建議 fps。',0.2,5,0.1,'fps'),
   roi_crop:B('ROI 裁切＋縮小','只把 ROI（加邊界）裁下來並縮小後再偵測，偵測框會換算回原畫面座標。'),
   crop_margin:S('裁切邊界','ROI 四周多保留的範圍（畫面比例）。',0,0.5,0.01),
   max_width:S('偵測最大寬度','送進偵測器的影像最大寬度。越小越快，但遠處的小臉可能偵測不到。',96,1920,16,'px')}]);
GROUPS.push(['畫面與聲音（瀏覽器端）',{fps:S('每秒送出張數','每秒送幾張畫面到伺服器。越高越即時，但越耗頻寬和 CPU。',0.5,5,0.5,'fps'),
   vad_k:S('VAD 靈敏度（門檻倍數）','音量超過「噪音底 × 這個倍數」才算說話。越小越靈敏。',1.2,8,0.1,'×'),
   vad_min:S('最低說話音量','絕對音量門檻（RMS），環境吵時調高。',0.001,0.05,0.001),
   vad_hang:S('說話尾音保留','音量變小後還要維持「說話中」多久，避免句子中間斷掉。',100,1500,50,'ms')}]);
GROUPS.push(['乾淨音軌（PureSound）',{audio_pipeline:{l:'音訊管線',h:'瀏覽器 VAD：用瀏覽器端的音量 VAD 判斷說話（目前預設）。PureSound：麥克風音訊（16 kHz PCM）串流到伺服器，用 PureSound 的串流人聲分離（dpcrn_curriculum_v2）去掉背景聲音，再用「乾淨音軌」判斷說話。需要伺服器安裝 puresound 套件與模型；沒有的話會在「乾淨音軌」面板顯示原因，並繼續用瀏覽器 VAD。',opts:{energy_vad:'瀏覽器 VAD',puresound:'PureSound 人聲分離'}},
   onset_entry_only:B('只在「進場」時開始對話','開啟時（PureSound 模式），對話只會在「乾淨音軌偵測到說話開頭」而且「畫面中有一位靠近的人」同時發生時開始；只有人經過、或只有聲音（電視、遠處的人）都不會開始。'),
   onset_needs_vision:B('說話需搭配畫面','PureSound 模式下，說話要「開始對話」或「從 AWAY 回來」時，最近 1.5 秒內畫面中必須有一位靠近的人（影像不確定時例外，改成只看聲音）。'),
   iso_onset_guard:B('起音保護（onset guard）','PureSound 內建：每段說話的前 1 秒先保留原始聲音，確認有人持續說話後才換成分離後的聲音，減少開頭字被吃掉。'),
   iso_reset_on_change:B('換人時重置','NEW_GROUP、新的對話或參與者換人時，重置人聲分離的串流狀態（也重新啟動起音保護）。'),
   clean_vad_margin_db:S('乾淨 VAD 門檻','分離後的聲音比噪音底高出多少 dB 才算說話。越小越靈敏。',3,30,0.5,'dB'),
   clean_vad_min_db:S('乾淨 VAD 最低音量','分離後的聲音至少要多大聲（dBFS）才算說話。遠處或很小聲的說話會被忽略。',-80,-10,1,'dBFS'),
   clean_vad_hang:S('乾淨 VAD 尾音保留','分離後的聲音變小後，還要維持「說話中」多久。',0,2,0.05,'s'),
   xcheck_enabled:B('影像交叉確認','PureSound 一直在過濾（收音範圍約 1 m），每 20 ms 用畫面確認一次結果：有乾淨說話但畫面沒人＝漏音（壓低、不算說話）；有乾淨說話但所有人都在近距離範圍外＝聲學誤判-超出範圍（壓低、不算說話）；乾淨音軌沒聲音，但範圍內有人而且原始麥克風有說話＝聲學誤判-漏收（多放一些原始聲音、算說話）。畫面不確定時只看聲音。'),
   near_range:S('近距離範圍','PureSound 的收音範圍。距離估計開啟時，所有人都比這個遠就視為「超出範圍」。沒開距離估計時只看 ROI 內有沒有人。',0.2,10,0.1,'m'),
   xcheck_hold:S('畫面保持時間','畫面每秒只有 0.5–2 張，聲音每 20 ms 一塊；看到人之後保持這麼久，對齊兩者、避免句首被切掉。',0,5,0.05,'s'),
   xcheck_stale:S('畫面過期時間','超過這麼久沒有新的畫面結果，就只看聲音。',0.2,30,0.1,'s'),
   xcheck_require_calib:B('距離需先校正','開啟時，要先在「距離」做過校正才會判斷「超出範圍」。'),
   xcheck_floor_db:S('壓低音量','漏音或超出範圍時輸出降低多少（≤ -60 dB 為靜音）。',-80,0,1,'dB'),
   xcheck_recover_mix:S('漏收時的增強比例','聲學誤判-漏收時，輸出中分離後聲音的比例；其餘是（時間對齊的）原始聲音。越小保留越多原聲。',0,0.9,0.05),
   xcheck_attack:S('音量上升時間','壓低解除時的淡入時間常數。',0.001,0.5,0.001,'s'),
   xcheck_release:S('音量下降時間','開始壓低時的淡出時間常數。',0.005,2,0.005,'s')}]);
const T=(l,h)=>({l,h,text:true});
GROUPS.push(['語音辨識（雲端，選用）',{asr_provider:{l:'語音辨識服務',h:'無：不做語音辨識（預設）。AssemblyAI：把聲音即時送到 AssemblyAI 雲端（Universal-Streaming v3）轉成文字。⚠ 這是外部雲端服務，聲音會離開這台裝置；請確認使用者同意與當地法規。需要 PureSound 管線與 API key。',opts:{none:'無',assemblyai:'AssemblyAI（雲端）'}},
   asr_source:{l:'送出的聲音',h:'乾淨：送出經過 PureSound 與影像確認後的乾淨音軌（預設）。原始：送出麥克風原始聲音。',opts:{clean:'乾淨音軌',raw:'原始'}},
   asr_feed:{l:'送出時機',h:'只在說話時：只有「校正後算說話」的片段送出聲音，其餘送靜音（保持時間對齊，內容不離開裝置）。整個對話：對話進行中的聲音全部送出。對話結束（ENDED）一律斷線。',opts:{speech:'只在說話時',session:'整個對話'}},
   asr_region:{l:'資料區域',h:'AssemblyAI 端點：全球（最近的區域）、美國或歐盟（資料不離開該區域）。',opts:{global:'全球',us:'美國',eu:'歐盟'}},
   asr_speech_model:{l:'模型',h:'Universal-3.6 Pro：32 種語言（含中文 zh、粵語 yue），可混用語言（預設）。Universal-3.5 Pro：19 種語言（含 zh，不含 yue）。Universal-Streaming English：只有英文。Universal-Streaming Multilingual：英、西、德、法、葡、義，不含中文。',opts:{'universal-3-6-pro':'Universal-3.6 Pro','universal-3-5-pro':'Universal-3.5 Pro','universal-streaming-english':'Universal-Streaming English','universal-streaming-multilingual':'Universal-Streaming Multilingual'}},
   asr_language_codes:T('語言','逗號分隔的語言代碼（例如 zh,en），偏向這些語言但仍可混用；留空＝不限制。只適用 Universal-3.x Pro。'),
   asr_keyterms:T('關鍵詞','逗號分隔，最多 100 個，提高這些詞的辨識率（例如人名、產品名）。'),
   asr_format_turns:B('格式化結果','句子加標點與大小寫。只適用 Universal-Streaming 模型（3.x Pro 一律格式化）。'),
   asr_eot_confidence:S('句尾信心門檻','判斷一句話結束的信心門檻。只適用 Universal-Streaming 模型。',0,1,0.05),
   asr_min_turn_silence:S('最短句尾靜默','靜默多久開始檢查是否句尾。',50,10000,50,'ms'),
   asr_max_turn_silence:S('最長句尾靜默','靜默這麼久一定結束這句話。',100,10000,50,'ms')}]);
GROUPS.push(['跨 session 使用者判斷（雲端，選用）',{reid_enabled:B('跨 session 使用者判斷','新的 session 開始時，把這次第一張清楚的臉和上一位（最近幾位）使用者的參考臉送到雲端比對：同一人就接續上一段對話（沿用 conversation_id），不同人或不確定就開新對話。session 進行中不會呼叫；沒有保存中的參考臉（第一次使用或已過期）就直接開新對話，不呼叫。⚠ 外部雲端服務：人臉裁切會離開這台裝置。'),
   reid_consent:B('我已告知使用者並取得同意','必須勾選才會運作：參考臉（每位使用者一張臉部裁切）只保存在伺服器記憶體、最多保存期限內，並會送到第三方雲端比對。不寫入磁碟或 log；只有開啟「顯示比對照片」時，才會把小縮圖送到這個瀏覽器顯示。'),
   reid_show_crops:B('顯示比對照片','在「使用者比對」卡片顯示這次送去比對的臉和比對到的參考臉（小縮圖，約 112 px）。縮圖只經這個連線送到這個瀏覽器顯示，不寫入磁碟、log、除錯事件或錄音；「清除參考臉」或重設時一併清除。'),
   reid_provider:{l:'雲端服務',h:'比對用的雲端後端（可抽換）。目前提供 Jev（Typesafe）。',opts:{jev:'Jev（Typesafe）'}},
   reid_model:{l:'模型',h:'jev-latest 會跟著服務更新；固定 jev-1.13.0 可確保結果可重現。',opts:{'jev-latest':'jev-latest','jev-1.13.0':'jev-1.13.0'}},
   reid_threshold:S('同一人門檻','分數達到這個值才視為同一人並接續對話。認錯人的代價較高，預設 0.9（LFW 評測：FAR 0%、FRR 14%）。',0.5,0.99,0.01),
   reid_uncertain_low:S('不確定下限','分數介於這個值和門檻之間時，再取一張新畫面比對一次；仍未達門檻就開新對話。',0,0.95,0.05),
   reid_ttl:S('參考臉保存時間','上一位使用者的參考臉在記憶體中保存多久，到期自動刪除（最長 10 分鐘）。',30,600,30,'s'),
   reid_max_refs:S('比對最近幾位','保存並比對最近幾位使用者（最新的先比，命中就停）。每位比對都算一次呼叫。',1,5,1,'位'),
   reid_min_face:S('最小臉寬','臉的寬度至少佔畫面這個比例才拿來比對。',0.02,0.5,0.01),
   reid_face_wait:S('等待清楚的臉','第一位參與者出現後，最多等這麼久拿到一張清楚的臉；等不到就開新對話。',0.5,15,0.5,'s'),
   reid_decision_timeout:S('最長等待時間','新 session 開始後最多等這麼久（MAYBE_AWAY／AWAY 期間暫停計時）；超過就開新對話。',0.5,30,0.5,'s'),
   reid_timeout:S('單次呼叫逾時','雲端呼叫超過這個時間就放棄並開新對話。',0.3,10,0.1,'s'),
   reid_cooldown:S('冷卻時間','兩次判斷之間至少間隔這麼久；session 快速重開時直接開新對話，不呼叫。',0,120,1,'s'),
   reid_cap_per_hour:S('每小時上限','每小時最多呼叫次數，達到後一律開新對話。',1,600,1,'次'),
   reid_cap_total:S('每連線上限','這個連線最多呼叫次數。',1,10000,1,'次'),
   reid_cost_per_call:S('每次呼叫單價','自行填入的單價，只用來估算成本（0＝只顯示 tokens）。',0,10,0.0001)}]);
const META=Object.assign({},...GROUPS.map(g=>g[1]));
// 推薦組合＝實測（低角度機器人鏡頭、單人近距離對話）調好的整套設定；情境組合都以它為基礎再微調
const REC={tracker_mode:'kalman',presence_model:'hmm',intent_model:'score',pose_enabled:true,distance_enabled:true,
 audio_pipeline:'puresound',xcheck_enabled:true,near_range:1.6,min_conf:0.4,leave_halflife:1.5,force_detect_seconds:2,
 join_seconds:1.5,grace_seconds:10,face_roi:[0.25,0.1,0.8,0.89],pose_roi:[0.18,0.04,0.85,0.92]};
const PRESETS={
 lowcam:{...REC,min_conf:0.35,leave_halflife:2,force_detect_seconds:1.5,face_roi:[0.2,0.05,0.85,0.95],pose_roi:[0.1,0,0.9,1],body_min_conf:0.35},
 single:{...REC,face_roi:[0.25,0.05,0.75,0.95],present_timeout:45,empty_timeout:5,new_group_seconds:3,absent_frames:3},
 group:{...REC,face_roi:[0.08,0.02,0.92,0.98],join_seconds:3,grace_seconds:20,present_frames:2,absent_frames:4,present_timeout:90,empty_timeout:8,new_group_seconds:5,cos_match:0.42,fps:2,near_range:2},
 noisy:{...REC,priority_mode:'vision',vad_k:3.5,vad_min:0.02,vad_hang:300,min_conf:0.5,near_range:1.2,base_timeout:15,present_timeout:45,empty_timeout:5}};
const KEY='presence-cfg-v1';
const migrate=o=>{o={...o};if(!o._v091){if(o.max_width===320)o.max_width=640;o._v091=1}if(o.roi&&!o.face_roi)o.face_roi=o.roi;delete o.roi;return o};
let cfg=Object.assign(structuredClone({...DEF,_v091:1}),migrate(JSON.parse(localStorage.getItem(KEY)||'{}')));cfg.monitor='off';// 監聽每次都預設關閉（避免回授）
const save=()=>localStorage.setItem(KEY,JSON.stringify(cfg));
const toast=t=>{const e=$('toast');e.textContent=t;e.classList.add('is-visible');clearTimeout(e._t);e._t=setTimeout(()=>e.classList.remove('is-visible'),2200)};
const clamp=(k,v)=>{const m=META[k];if(m.text)return String(v).slice(0,2000);if(m.bool)return v===true||v==='true'||v===1;if(m.opts)return m.opts[v]?v:DEF[k];return Math.min(m.max,Math.max(m.min,Number(v)))};
// ---------- settings: modules in data-flow order (v0.10) ----------
// Each module: enable/mode key at the top, then parameters. `adv` = shown only in 進階 mode.
// Rules per key: show(c) -> hidden when false (irrelevant), need:[fn,reason] -> shown disabled with the reason.
const ON=k=>c=>!!c[k],EQ=(k,v)=>c=>c[k]===v,PS=c=>c.audio_pipeline==='puresound',DIST=c=>!!c.distance_enabled;
const MODULES=[
 {id:'input',name:'輸入（鏡頭／麥克風）',keys:['fps','vad_k','vad_min','vad_hang'],adv:['vad_min','vad_hang'],
  rules:{fps:{need:[c=>!c.adaptive_fps,'「效能 → 自適應 fps」開啟時由伺服器決定送幀速率']},vad_k:{need:[c=>!PS(c),'使用 PureSound 時改用乾淨音軌的 VAD（見「音訊」）']},vad_min:{need:[c=>!PS(c),'使用 PureSound 時不使用']},vad_hang:{need:[c=>!PS(c),'使用 PureSound 時不使用']}},
  status:c=>`${c.adaptive_fps?'自適應':c.fps+' fps'} · ${PS(c)?'PCM 串流':'瀏覽器 VAD'}`},
 {id:'detect',name:'視覺偵測',keys:['face_roi','min_conf','embed_backend','npu_precision','cos_match_int8'],adv:['cos_match_int8','npu_precision'],
  rules:{npu_precision:{show:EQ('embed_backend','npu')},cos_match_int8:{show:c=>c.embed_backend!=='fp32'}},status:c=>`YuNet · ${c.embed_backend}`},
 {id:'pose',name:'姿勢（選配）',gate:'pose_enabled',keys:['pose_conf','body_min_conf','pose_roi','roi_rule','turned_away_early','turned_away_seconds','recede_ratio'],adv:['body_min_conf','recede_ratio','turned_away_seconds'],
  rules:{turned_away_seconds:{show:ON('turned_away_early')}},status:c=>c.pose_enabled?'開':'關'},
 {id:'distance',name:'距離（選配）',gate:'distance_enabled',keys:['camera_hfov','distance_calib','zone_min','zone_max','leave_distance','face_width_m','shoulder_width_m','distance_ema'],adv:['face_width_m','shoulder_width_m','distance_ema'],
  rules:{shoulder_width_m:{need:[ON('pose_enabled'),'肩寬需要開啟「姿勢」']}},status:c=>c.distance_enabled?'開':'關'},
 {id:'track',name:'追蹤',gate:'tracker_mode',keys:['join_seconds','grace_seconds','cos_match','cos_weak','iou_match','embed_ema','coast_seconds','coast_present_conf','gate_mahalanobis','appearance_min_sim','appearance_ema'],
  adv:['cos_weak','iou_match','embed_ema','gate_mahalanobis','appearance_min_sim','appearance_ema','coast_present_conf'],
  rules:Object.fromEntries(['coast_seconds','coast_present_conf','gate_mahalanobis','appearance_min_sim','appearance_ema'].map(k=>[k,{show:EQ('tracker_mode','kalman')}])),status:c=>c.tracker_mode==='kalman'?'Kalman':'傳統'},
 {id:'intent',name:'互動意圖',gate:'intent_model',keys:['join_threshold','drop_threshold','drop_seconds','intent_tau','intent_weighted_presence','w_facing','w_distance','w_dwell','w_approach','w_still','w_center'],
  adv:['intent_tau','w_facing','w_distance','w_dwell','w_approach','w_still','w_center','drop_seconds'],
  rules:{...Object.fromEntries(['join_threshold','drop_threshold','drop_seconds','intent_tau','intent_weighted_presence','w_facing','w_distance','w_dwell','w_approach','w_still','w_center'].map(k=>[k,{show:EQ('intent_model','score')}])),
   w_distance:{show:EQ('intent_model','score'),need:[DIST,'需要開啟「距離」；關閉時這個權重自動忽略']},intent_weighted_presence:{show:EQ('intent_model','score'),need:[EQ('presence_model','hmm'),'只在「在場判斷 → 機率」模式有效']}},
  status:c=>c.intent_model==='score'?'意圖分數':'規則'},
 {id:'presence',name:'在場判斷',gate:'presence_model',keys:['priority_mode','speech_join','present_frames','absent_frames','leave_halflife','detect_gain','lowconf_factor','p_active','p_maybe','p_away','away_hold_seconds'],
  adv:['detect_gain','lowconf_factor','away_hold_seconds','p_active','speech_join'],
  rules:{...Object.fromEntries(['leave_halflife','detect_gain','lowconf_factor','p_active','p_maybe','p_away','away_hold_seconds'].map(k=>[k,{show:EQ('presence_model','hmm')}])),present_frames:{show:EQ('presence_model','rules')},absent_frames:{show:EQ('presence_model','rules')}},
  status:c=>`${c.presence_model==='hmm'?'機率':'規則'} · ${{vision:'影像優先',balanced:'平衡',audio:'聲音優先'}[c.priority_mode]}`},
 {id:'state',name:'狀態機／逾時',keys:['idle_after','present_timeout','empty_timeout','base_timeout','present_max_silence','maybe_away_frac','end_after_away','new_group_seconds'],adv:['maybe_away_frac','base_timeout'],
  rules:{present_max_silence:{need:[EQ('priority_mode','vision'),'只在「影像優先」使用']},present_timeout:{need:[c=>c.priority_mode!=='vision','「影像優先」時改用「有人時最長靜默」']}},status:c=>`逾時 ${c.present_timeout}/${c.empty_timeout} s`},
 {id:'audio',name:'音訊（PureSound＋影像交叉確認）',gate:'audio_pipeline',keys:['onset_entry_only','onset_needs_vision','iso_onset_guard','iso_reset_on_change','clean_vad_margin_db','clean_vad_min_db','clean_vad_hang','xcheck_enabled','near_range','xcheck_hold','xcheck_stale','xcheck_require_calib','xcheck_floor_db','xcheck_recover_mix','xcheck_attack','xcheck_release'],
  adv:['iso_reset_on_change','clean_vad_hang','xcheck_stale','xcheck_attack','xcheck_release','xcheck_recover_mix','clean_vad_min_db'],
  rules:{...Object.fromEntries(['onset_entry_only','onset_needs_vision','iso_onset_guard','iso_reset_on_change','clean_vad_margin_db','clean_vad_min_db','clean_vad_hang','xcheck_enabled'].map(k=>[k,{show:PS}])),
   ...Object.fromEntries(['near_range','xcheck_hold','xcheck_stale','xcheck_floor_db','xcheck_recover_mix','xcheck_attack','xcheck_release'].map(k=>[k,{show:c=>PS(c)&&c.xcheck_enabled}])),
   near_range:{show:c=>PS(c)&&c.xcheck_enabled,need:[DIST,'情況 3「聲學誤判-超出範圍」需要距離估計；「距離」關閉時跳過，只檢查 ROI 內有沒有人']},
   xcheck_require_calib:{show:c=>PS(c)&&c.xcheck_enabled,need:[DIST,'需要開啟「距離」']}},
  status:c=>PS(c)?`PureSound${c.xcheck_enabled?' + 交叉確認':''}`:'瀏覽器 VAD'},
 {id:'asr',name:'語音辨識（雲端，選用）',gate:'asr_provider',keys:['asr_source','asr_feed','asr_region','asr_speech_model','asr_language_codes','asr_keyterms','asr_format_turns','asr_eot_confidence','asr_min_turn_silence','asr_max_turn_silence'],
  adv:['asr_eot_confidence','asr_min_turn_silence','asr_max_turn_silence','asr_keyterms','asr_format_turns'],
  rules:{...Object.fromEntries(['asr_source','asr_feed','asr_region','asr_speech_model','asr_language_codes','asr_keyterms','asr_min_turn_silence','asr_max_turn_silence'].map(k=>[k,{show:EQ('asr_provider','assemblyai')}])),
   asr_language_codes:{show:c=>c.asr_provider==='assemblyai'&&c.asr_speech_model.endsWith('-pro')},
   asr_format_turns:{show:c=>c.asr_provider==='assemblyai'&&c.asr_speech_model.startsWith('universal-streaming')},asr_eot_confidence:{show:c=>c.asr_provider==='assemblyai'&&c.asr_speech_model.startsWith('universal-streaming')}},
  gateNeed:[PS,'需要「音訊」模組使用 PureSound 管線'],status:c=>c.asr_provider==='none'?'關':(PS(c)?'AssemblyAI':'AssemblyAI（需要 PureSound）')},
 {id:'reid',name:'跨 session 使用者判斷（雲端，選用）',gate:'reid_enabled',keys:['reid_consent','reid_show_crops','reid_provider','reid_model','reid_threshold','reid_uncertain_low','reid_ttl','reid_max_refs','reid_min_face','reid_face_wait','reid_decision_timeout','reid_timeout','reid_cooldown','reid_cap_per_hour','reid_cap_total','reid_cost_per_call'],
  adv:['reid_provider','reid_uncertain_low','reid_min_face','reid_face_wait','reid_decision_timeout','reid_timeout','reid_cooldown','reid_cap_total'],rules:{},
  status:c=>!c.reid_enabled?'關':c.reid_consent?'開':'未同意'},
 {id:'perf',name:'效能',keys:['motion_gate','motion_threshold','force_detect_seconds','embed_throttle','embed_interval','embed_move_thresh','adaptive_fps','fps_low','fps_high','roi_crop','crop_margin','max_width'],
  adv:['motion_threshold','force_detect_seconds','embed_interval','embed_move_thresh','crop_margin'],
  rules:{motion_threshold:{show:ON('motion_gate')},force_detect_seconds:{show:ON('motion_gate')},embed_interval:{show:ON('embed_throttle')},embed_move_thresh:{show:ON('embed_throttle')},fps_low:{show:ON('adaptive_fps')},fps_high:{show:ON('adaptive_fps')},crop_margin:{show:ON('roi_crop')}},
  status:c=>`閘門${c.motion_gate?'開':'關'} · ${c.max_width}px`},
 {id:'display',name:'顯示（只影響這個瀏覽器）',keys:['mirror_view','show_face_roi','show_pose_roi'],adv:[],rules:{show_pose_roi:{show:ON('pose_enabled')}},status:c=>c.mirror_view?'鏡像':'原始'}];
const MOD_PRESETS={
 recommended:{...REC},
 lowpower:{motion_gate:true,embed_throttle:true,embed_interval:5,adaptive_fps:true,fps_low:0.5,fps_high:1,roi_crop:true,max_width:480,pose_enabled:false,distance_enabled:false}};
let ADV=localStorage.getItem('presence-adv')==='1';
function ruleOf(m,k){return(m.rules||{})[k]||{}}
function buildUI(){
 const all=new Set(MODULES.flatMap(m=>[m.gate,...m.keys].filter(Boolean)));const miss=Object.keys(META).filter(k=>!all.has(k));if(miss.length)console.error('settings keys without module:',miss);
 $('flow').innerHTML=MODULES.map((m,i)=>`${i?'<span class="arr">→</span>':''}<button type="button" class="chip" data-mod="${m.id}"><b>${m.name.replace(/（.*/,'')}</b><small id="fs_${m.id}"></small></button>`).join('');
 $('flow').querySelectorAll('[data-mod]').forEach(b=>b.onclick=()=>{const d=$('mod_'+b.dataset.mod);d.open=true;d.scrollIntoView({behavior:'smooth',block:'start'})});
 $('groups').innerHTML=MODULES.map(m=>`<details class="mod" id="mod_${m.id}" ${m.id==='detect'?'open':''}><summary>${m.name} <span class="mst" id="ms_${m.id}"></span><button type="button" class="button button-ghost button-small mreset" data-mreset="${m.id}" title="這個模組恢復預設值">重設</button></summary>`
  +(m.gate?`<div class="gate">${rowHTML(m.gate)}<div class="gnote" id="gn_${m.id}"></div></div>`:'')+`<div class="mbody">`+m.keys.map(rowHTML).join('')+'</div></details>').join('');
 $('groups').querySelectorAll('[data-mreset]').forEach(b=>b.onclick=e=>{e.preventDefault();e.stopPropagation();const m=MODULES.find(x=>x.id===b.dataset.mreset);
  const part={};[m.gate,...m.keys].filter(Boolean).forEach(k=>part[k]=structuredClone(DEF[k]));setCfg(part);restartFrames();toast('已重設：'+m.name)});
 $('adv').checked=ADV;$('adv').onchange=()=>{ADV=$('adv').checked;localStorage.setItem('presence-adv',ADV?'1':'0');applyDeps()};
 $('groups').querySelectorAll('input,select').forEach(inp=>{
  const ev=inp.type==='range'?'input':'change';
  if(inp.type==='checkbox'){inp.addEventListener('change',()=>setCfg({[inp.dataset.k]:inp.checked}));return}
  if(inp.type==='text'){inp.addEventListener('change',()=>setCfg({[inp.dataset.k]:inp.value}));return}
  if(inp.tagName==='SELECT'){inp.addEventListener('change',()=>setCfg({[inp.dataset.k]:inp.value}));return}
  inp.addEventListener(ev,()=>{const k=inp.dataset.k;
   if(BOXES.includes(k)){const r=[...cfg[k]];r[+inp.dataset.i]=Math.min(1,Math.max(0,+inp.value));setCfg({[k]:r})}
   else setCfg({[k]:clamp(k,inp.value)},inp)});
  if(inp.type==='range')inp.addEventListener('change',()=>sendCfg());
 });
 syncUI();
}
function rowHTML(k){const m=META[k];const d=Array.isArray(DEF[k])?DEF[k].join(', '):DEF[k];
 const tip=`<div class="why" id="why_${k}"></div><p class="tip">${m.h}　<span class="def">預設 ${m.bool?(d?'開':'關'):m.opts?m.opts[d]:d}${m.u?' '+m.u:''}</span></p>`;
 if(m.bool)return `<div class="row" id="row_${k}"><label class="toggle-row" style="margin:0" title="${m.h}"><input type="checkbox" data-k="${k}"><span class="toggle-ui"></span><span>${m.l}</span></label>${tip}</div>`;
 if(m.text)return `<div class="row field" id="row_${k}"><label for="f_${k}">${m.l}</label><input class="control" id="f_${k}" type="text" data-k="${k}">${tip}</div>`;
 if(m.box)return `<div class="row" id="row_${k}"><div class="top"><label title="${m.h}">${m.l}</label></div><div class="roi4">${['x0','y0','x1','y1'].map((n,j)=>`<input class="control" type="number" data-k="${k}" data-i="${j}" min="0" max="1" step="0.01" title="${n}">`).join('')}</div>${tip}</div>`;
 if(m.opts)return `<div class="row field" id="row_${k}"><label for="f_${k}">${m.l}</label><select class="control" id="f_${k}" data-k="${k}">${Object.entries(m.opts).map(([o,t])=>`<option value="${o}">${t}</option>`).join('')}</select>${tip}</div>`;
 return `<div class="row" id="row_${k}"><div class="field-label range-label"><label for="f_${k}">${m.l}</label><output id="o_${k}"></output></div><input class="range" id="f_${k}" type="range" data-k="${k}" min="${m.min}" max="${m.max}" step="${m.step}"><div class="top" style="margin-top:6px"><input class="control" type="number" data-k="${k}" min="${m.min}" max="${m.max}" step="${m.step}" style="width:84px"><span class="def">${m.u}</span></div>${tip}</div>`}
function applyDeps(){for(const m of MODULES){
  const gateOff=m.gate&&(META[m.gate].bool?!cfg[m.gate]:false);
  for(const k of m.keys){const r=$('row_'+k);if(!r)continue;const ru=ruleOf(m,k);
   const vis=!gateOff&&(!ru.show||ru.show(cfg))&&(ADV||!m.adv.includes(k)||r.classList.contains('changed'));
   const ok=!ru.need||ru.need[0](cfg);r.hidden=!vis;r.classList.toggle('dis',!ok);r.querySelectorAll('input,select').forEach(x=>x.disabled=!ok);
   $('why_'+k).textContent=ok?'':'⛔ '+ru.need[1]}
  const st=m.status(cfg),gn=$('gn_'+m.id);$('ms_'+m.id).textContent=st;const fs=$('fs_'+m.id);if(fs){fs.textContent=st;fs.parentElement.classList.toggle('off',/^(關|瀏覽器 VAD)$/.test(st))}
  if(gn){const bad=m.gateNeed&&!m.gateNeed[0](cfg)&&cfg[m.gate]!=='none';gn.textContent=bad?'⚠ '+m.gateNeed[1]:gateOff?'（關閉：下方參數不適用）':'';}
  const body=$('mod_'+m.id).querySelector('.mbody'),hid=m.keys.filter(k=>$('row_'+k)&&$('row_'+k).hidden).length;
  body.dataset.hidden=hid;}}
function syncUI(skip){if(cfg.asr_provider==='none')captionOff();
 document.querySelectorAll('#groups input,#groups select').forEach(inp=>{if(inp===skip)return;const k=inp.dataset.k;
  if(inp.type==='checkbox'){inp.checked=!!cfg[k];return}
  const v=BOXES.includes(k)?(+cfg[k][+inp.dataset.i]).toFixed(2):cfg[k];inp.value=v;
  if(inp.type==='range'){const o=$('o_'+k);if(o)o.textContent=META[k].u?`${v} ${META[k].u}`:v}});
 for(const k in DEF){const r=$('row_'+k);if(r)r.classList.toggle('changed',JSON.stringify(cfg[k])!==JSON.stringify(DEF[k]))}
 if(typeof applyDeps==='function'&&$('flow').childElementCount)applyDeps();
}
let sendTimer;
function setCfg(part,src,immediate){Object.assign(cfg,part);save();syncUI(src);
 if(src&&src.type==='range'){clearTimeout(sendTimer);sendTimer=setTimeout(sendCfg,250)}else sendCfg();
 if('fps' in part)restartFrames();draw();}
function serverCfg(){const o={};for(const k in DEF)if(!CLIENT.includes(k))o[k]=cfg[k];return o}
function sendCfg(){if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'config',config:serverCfg()}))}
document.querySelectorAll('#presets [data-p]').forEach(b=>b.onclick=()=>{const p=b.dataset.p;
 cfg=Object.assign(structuredClone(DEF),p==='reset'?{}:(PRESETS[p]||MOD_PRESETS[p]));save();syncUI();sendCfg();restartFrames();draw();toast('已套用：'+b.textContent)});
$('exp').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(cfg,null,2)],{type:'application/json'}));a.download='presence-config.json';a.click()};
$('impB').onclick=()=>$('imp').click();
$('imp').onchange=async e=>{try{const j=JSON.parse(await e.target.files[0].text());const n=structuredClone(DEF);
 const jm=migrate(j);for(const k in DEF)if(k in jm)n[k]=BOXES.includes(k)?jm[k].slice(0,4).map(x=>Math.min(1,Math.max(0,+x))):clamp(k,jm[k]);cfg=n;save();syncUI();sendCfg();restartFrames();draw();toast('已匯入設定')}
 catch(err){toast('匯入失敗：'+err.message)}e.target.value=''};
const drawer=(o)=>{$('drawer').classList.toggle('is-open',o);$('scrim').classList.toggle('is-open',o);document.body.classList.toggle('has-drawer',o)};
$('openSet').onclick=()=>drawer(true);$('closeSet').onclick=$('scrim').onclick=()=>drawer(false);
