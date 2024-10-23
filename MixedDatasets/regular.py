import re

log_text = '''
Args in experiment:
Namespace(activation='gelu', batch_size=16, c_out=862, channel_independence=False, checkpoints='./checkpoints/', class_strategy='projection', cut_freq=48, d_ff=512, d_layers=1, d_model=512, data='custom', data_path='traffic.csv', datasetsDict_list=[('custom', {'embed': 'timeF', 'freq': 'h', 'root_path': './dataset/traffic/', 'data_path': 'traffic.csv', 'seq_len': 96, 'label_len': 48, 'pred_len': 96, 'features': 'M', 'target': 'OT', 'num_workers': 0})], dec_in=862, des='Exp', devices='0,1,2,3', distil=True, do_predict=False, dropout=0.1, e_layers=4, efficient_training=False, embed='timeF', enc_in=862, exp_name='MTSF_wMask', factor=1, features='M', freq='h', gpu=0, inverse=False, is_training=0, itr=1, label_len=48, learning_rate=0.0002, loss='MSE', lradj='type1', mask_ratio=0.3, model='iTransformer_wMask', model_id='traffic_192_96_wMask', model_load_from='/root/iTransformer/checkpoints/traffic_192_96_wMask_wFreqMaskLoss_30maskratio0.2reconweight_sameprojector_______2024022412____________iTransformer_wMask_custom_M_ft192_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth', moving_avg=25, n_heads=8, num_workers=10, output_attention=False, partial_start_index=0, patience=3, pred_len=96, recon_loss_weight=0.2, root_path='/root/autodl-tmp/iTransformer_datasets/traffic/', seq_len=192, target='OT', target_data_path='electricity.csv', target_root_path='./data/electricity/', train_epochs=12, use_amp=False, use_gpu=True, use_multi_gpu=False, use_norm=True)
Use GPU: cuda:0
Model size: 7.070432M
>>>>>>>testing : traffic_192_96_wMask_iTransformer_wMask_custom_M_ft192_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 3413
loading model
No trained model found. Using originally intialized model from /root/iTransformer/checkpoints/traffic_192_96_wMask_wFreqMaskLoss_30maskratio0.2reconweight_sameprojector_______2024022412____________iTransformer_wMask_custom_M_ft192_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3255765438079834
current mse: 0.28989970684051514
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3455882668495178
current mse: 0.29025813937187195
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3516220450401306
current mse: 0.29538801312446594
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.41010212898254395
current mse: 0.2981514036655426
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3082941770553589
current mse: 0.2992987334728241
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3730069398880005
current mse: 0.3027379512786865
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3419356942176819
current mse: 0.2935134470462799
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.5643107891082764
current mse: 0.30565810203552246
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33664584159851074
current mse: 0.29912182688713074
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34537264704704285
current mse: 0.30249518156051636
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3110693097114563
current mse: 0.3033961355686188
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30663204193115234
current mse: 0.2923484146595001
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2992953360080719
current mse: 0.28231555223464966
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2989824712276459
current mse: 0.2809954583644867
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4988496005535126
current mse: 0.2888002395629883
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.5427427887916565
current mse: 0.28182923793792725
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3645303249359131
current mse: 0.26781272888183594
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.372177392244339
current mse: 0.2612554728984833
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4229142963886261
current mse: 0.2742263376712799
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3216957151889801
current mse: 0.2770036458969116
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.345363587141037
current mse: 0.2890130877494812
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35549962520599365
current mse: 0.2995690703392029
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27292996644973755
current mse: 0.29121434688568115
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2926171123981476
current mse: 0.29516953229904175
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3128824830055237
current mse: 0.28615304827690125
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26277726888656616
current mse: 0.29756975173950195
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.373421311378479
current mse: 0.3059452474117279
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30001723766326904
current mse: 0.2980150878429413
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.504036545753479
current mse: 0.2977304458618164
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3046557605266571
current mse: 0.30107566714286804
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3650617301464081
current mse: 0.30301734805107117
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4346148669719696
current mse: 0.3053320646286011
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3936740458011627
current mse: 0.2974914312362671
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.45486119389533997
current mse: 0.3020663261413574
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3426065444946289
current mse: 0.3043152391910553
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.37469547986984253
current mse: 0.3038959801197052
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34458544850349426
current mse: 0.2806181311607361
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33222854137420654
current mse: 0.2763333022594452
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4152304232120514
current mse: 0.28181615471839905
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3270803689956665
current mse: 0.28717541694641113
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.371930330991745
current mse: 0.30172690749168396
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2937679588794708
current mse: 0.2927386462688446
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2971157431602478
current mse: 0.29416146874427795
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27674850821495056
current mse: 0.296993613243103
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3796881437301636
current mse: 0.3076200783252716
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28323063254356384
current mse: 0.3090690076351166
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32067957520484924
current mse: 0.3087306022644043
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27100643515586853
current mse: 0.31533703207969666
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28587400913238525
current mse: 0.31462588906288147
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3277930021286011
current mse: 0.31301671266555786
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3662843406200409
current mse: 0.3127923905849457
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.37608468532562256
current mse: 0.3217223882675171
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3686227798461914
current mse: 0.32959720492362976
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.49643078446388245
current mse: 0.32246536016464233
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32233887910842896
current mse: 0.3284885287284851
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3915599286556244
current mse: 0.33365243673324585
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29697278141975403
current mse: 0.33638113737106323
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2853817939758301
current mse: 0.3299163579940796
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.5212817788124084
current mse: 0.3340025544166565
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30749958753585815
current mse: 0.32448646426200867
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4098323583602905
current mse: 0.31366264820098877
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27299439907073975
current mse: 0.31661278009414673
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35123252868652344
current mse: 0.3177452087402344
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36391088366508484
current mse: 0.3158811032772064
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3250662684440613
current mse: 0.3205409348011017
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38425737619400024
current mse: 0.3227435350418091
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3106590807437897
current mse: 0.3295454680919647
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3389906585216522
current mse: 0.3392449617385864
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33777111768722534
current mse: 0.33968302607536316
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2559719383716583
current mse: 0.3291691243648529
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32869836688041687
current mse: 0.33836156129837036
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29236894845962524
current mse: 0.3168184757232666
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38868239521980286
current mse: 0.31509751081466675
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.291706919670105
current mse: 0.32713428139686584
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.43529796600341797
current mse: 0.3343822956085205
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.44458284974098206
current mse: 0.3399392068386078
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32374003529548645
current mse: 0.3379819989204407
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3952227234840393
current mse: 0.33818116784095764
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.42276430130004883
current mse: 0.3344500958919525
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4586659073829651
current mse: 0.3320983350276947
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31081265211105347
current mse: 0.3335069417953491
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2836512625217438
current mse: 0.3363264799118042
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3484111726284027
current mse: 0.32598409056663513
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2836041748523712
current mse: 0.3231135606765747
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34925031661987305
current mse: 0.32655012607574463
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34265509247779846
current mse: 0.33121931552886963
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3790913224220276
current mse: 0.3231199383735657
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28905627131462097
current mse: 0.3277498781681061
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2926872670650482
current mse: 0.32400137186050415
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.6006284356117249
current mse: 0.33556458353996277
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28743305802345276
current mse: 0.34541335701942444
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2880423367023468
current mse: 0.3537735044956207
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38795042037963867
current mse: 0.3400176763534546
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3480461835861206
current mse: 0.34051230549812317
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34851783514022827
current mse: 0.3455400764942169
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.362216591835022
current mse: 0.3348551094532013
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3009033203125
current mse: 0.3365820646286011
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3088652193546295
current mse: 0.33934643864631653
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3582186698913574
current mse: 0.34206661581993103
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34272176027297974
current mse: 0.34224143624305725
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35480087995529175
current mse: 0.34166112542152405
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.450265496969223
current mse: 0.3371310234069824
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3124943673610687
current mse: 0.3393007516860962
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28298258781433105
current mse: 0.3286924958229065
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30108287930488586
current mse: 0.33473747968673706
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34721359610557556
current mse: 0.3393308222293854
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35106438398361206
current mse: 0.32932502031326294
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.44024360179901123
current mse: 0.33588624000549316
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2990487813949585
current mse: 0.3454708755016327
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3856569826602936
current mse: 0.34378567337989807
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3661092519760132
current mse: 0.33680328726768494
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30077046155929565
current mse: 0.33343812823295593
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36482059955596924
current mse: 0.3283347189426422
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3094065487384796
current mse: 0.3395013213157654
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3194711208343506
current mse: 0.3204209804534912
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36148184537887573
current mse: 0.31901419162750244
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2489604651927948
current mse: 0.3181113600730896
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.455626517534256
current mse: 0.3126826286315918
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4239017963409424
current mse: 0.3160042464733124
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34758996963500977
current mse: 0.3215717673301697
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3238919675350189
current mse: 0.3230817914009094
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4861018657684326
current mse: 0.3145183324813843
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27217528223991394
current mse: 0.3172578811645508
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3534521758556366
current mse: 0.32881054282188416
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39831140637397766
current mse: 0.30649319291114807
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29779452085494995
current mse: 0.3034469485282898
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3874204456806183
current mse: 0.3006802797317505
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3622293472290039
current mse: 0.2982286214828491
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.41430577635765076
current mse: 0.2928156852722168
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.6100016832351685
current mse: 0.29038774967193604
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3230224549770355
current mse: 0.30053138732910156
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3179517388343811
current mse: 0.29679006338119507
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32281404733657837
current mse: 0.31238704919815063
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25656095147132874
current mse: 0.3240658640861511
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3311624526977539
current mse: 0.3141195774078369
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28212615847587585
current mse: 0.30988070368766785
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26968657970428467
current mse: 0.3170468211174011
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32943248748779297
current mse: 0.29321128129959106
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.41753798723220825
current mse: 0.28455308079719543
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30828723311424255
current mse: 0.2890137732028961
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3216193616390228
current mse: 0.3032551407814026
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2836543023586273
current mse: 0.30509525537490845
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.6308329105377197
current mse: 0.30382126569747925
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30670735239982605
current mse: 0.2980117201805115
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.357069730758667
current mse: 0.2900255620479584
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.40757057070732117
current mse: 0.2902339994907379
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.393812894821167
current mse: 0.3009057641029358
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.313042014837265
current mse: 0.28044435381889343
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.37372657656669617
current mse: 0.27717363834381104
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39765456318855286
current mse: 0.276849627494812
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3234061598777771
current mse: 0.28337588906288147
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31494754552841187
current mse: 0.304100900888443
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34731626510620117
current mse: 0.29036661982536316
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3022221624851227
current mse: 0.2830866575241089
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29905569553375244
current mse: 0.28592246770858765
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3322494626045227
current mse: 0.2991228699684143
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3190731108188629
current mse: 0.32087188959121704
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38657116889953613
current mse: 0.3177250027656555
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2866630554199219
current mse: 0.3058282434940338
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2600836455821991
current mse: 0.28676682710647583
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28545817732810974
current mse: 0.2858435809612274
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30475834012031555
current mse: 0.2776131331920624
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3165591061115265
current mse: 0.2843625545501709
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2825813591480255
current mse: 0.2782696485519409
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3092387616634369
current mse: 0.2876298129558563
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31277111172676086
current mse: 0.31648319959640503
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2966015934944153
current mse: 0.30619585514068604
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.290187269449234
current mse: 0.29019808769226074
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30590981245040894
current mse: 0.2956998348236084
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30801448225975037
current mse: 0.3027617335319519
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29180726408958435
current mse: 0.304538369178772
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30925479531288147
current mse: 0.32702621817588806
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.49448809027671814
current mse: 0.30082571506500244
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3195233941078186
current mse: 0.2963407337665558
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4805399179458618
current mse: 0.2805597186088562
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34112536907196045
current mse: 0.2805120646953583
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33793941140174866
current mse: 0.28368961811065674
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2900806665420532
current mse: 0.28283822536468506
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35522595047950745
current mse: 0.28707078099250793
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2987673282623291
current mse: 0.28997841477394104
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3804275095462799
current mse: 0.29616090655326843
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29873159527778625
current mse: 0.2782827317714691
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35111409425735474
current mse: 0.28098124265670776
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3754426836967468
current mse: 0.2850675582885742
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.5374830961227417
current mse: 0.29789265990257263
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3013971149921417
current mse: 0.2857232689857483
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2854516804218292
current mse: 0.27029550075531006
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28655099868774414
current mse: 0.2656036615371704
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33683985471725464
current mse: 0.27517205476760864
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33322298526763916
current mse: 0.271347314119339
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30629435181617737
current mse: 0.2733556926250458
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2667599320411682
current mse: 0.26118770241737366
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3190373182296753
current mse: 0.26081347465515137
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26229625940322876
current mse: 0.26238059997558594
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28424862027168274
current mse: 0.262558251619339
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32498642802238464
current mse: 0.2673618495464325
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3189542591571808
current mse: 0.26520100235939026
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3596101999282837
current mse: 0.2568818926811218
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29923519492149353
current mse: 0.25573399662971497
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34760764241218567
current mse: 0.2591125965118408
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3038882911205292
current mse: 0.2649368345737457
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3452553451061249
current mse: 0.2736031711101532
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30988144874572754
current mse: 0.2669333219528198
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27261289954185486
current mse: 0.25451284646987915
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28660866618156433
current mse: 0.2525661587715149
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29535093903541565
current mse: 0.25410276651382446
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32643768191337585
current mse: 0.2655634880065918
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28487417101860046
current mse: 0.2731558680534363
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31283921003341675
current mse: 0.27689582109451294
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.5085095167160034
current mse: 0.2674975097179413
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27626070380210876
current mse: 0.2752703130245209
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31136104464530945
current mse: 0.2835492491722107
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3694884479045868
current mse: 0.2966393232345581
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28676095604896545
current mse: 0.2974264919757843
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3232059180736542
current mse: 0.29126253724098206
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3085271716117859
current mse: 0.2948494851589203
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34069955348968506
current mse: 0.28249436616897583
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3802775740623474
current mse: 0.27942952513694763
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35426390171051025
current mse: 0.27608269453048706
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29148900508880615
current mse: 0.2795428931713104
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3348633646965027
current mse: 0.28706347942352295
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3554089665412903
current mse: 0.2754193842411041
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32493385672569275
current mse: 0.28101304173469543
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3170345425605774
current mse: 0.2816642224788666
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3283708989620209
current mse: 0.27893611788749695
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3385162055492401
current mse: 0.282956063747406
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2843126654624939
current mse: 0.30177071690559387
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3793497085571289
current mse: 0.2934192717075348
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2904057502746582
current mse: 0.2870602011680603
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26701483130455017
current mse: 0.27280518412590027
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2625739276409149
current mse: 0.27603840827941895
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3087180256843567
current mse: 0.27059292793273926
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2827126979827881
current mse: 0.2710595428943634
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2984870970249176
current mse: 0.27362334728240967
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2539881467819214
current mse: 0.27781811356544495
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26659509539604187
current mse: 0.2808707654476166
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27638867497444153
current mse: 0.29151451587677
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2519882619380951
current mse: 0.2799915373325348
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26246294379234314
current mse: 0.2613738775253296
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2507159411907196
current mse: 0.25510674715042114
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29986563324928284
current mse: 0.2532958984375
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26864033937454224
current mse: 0.24926619231700897
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.288082480430603
current mse: 0.2577536404132843
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36207249760627747
current mse: 0.26225563883781433
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33667808771133423
current mse: 0.2530888617038727
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.310563862323761
current mse: 0.2510581910610199
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3275185525417328
current mse: 0.24026617407798767
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3113320469856262
current mse: 0.25234901905059814
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3080288767814636
current mse: 0.2621941864490509
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4014038145542145
current mse: 0.24601708352565765
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.41113558411598206
current mse: 0.2532031536102295
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3082422912120819
current mse: 0.2522033154964447
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24668173491954803
current mse: 0.26650235056877136
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2781928479671478
current mse: 0.2752007842063904
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3232984244823456
current mse: 0.2875790596008301
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31373095512390137
current mse: 0.29344499111175537
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2680511176586151
current mse: 0.28549960255622864
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33164066076278687
current mse: 0.27271729707717896
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2994329333305359
current mse: 0.28159812092781067
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38480332493782043
current mse: 0.27965086698532104
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28136762976646423
current mse: 0.27524060010910034
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2606249451637268
current mse: 0.2753790616989136
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2601220905780792
current mse: 0.27251720428466797
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2742145359516144
current mse: 0.27291056513786316
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24681974947452545
current mse: 0.27793988585472107
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28499454259872437
current mse: 0.28256019949913025
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2980533838272095
current mse: 0.2862800061702728
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38481512665748596
current mse: 0.29239633679389954
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30577680468559265
current mse: 0.2820969521999359
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3300051689147949
current mse: 0.27985307574272156
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27800559997558594
current mse: 0.2802285850048065
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2731077969074249
current mse: 0.28376305103302
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30204081535339355
current mse: 0.27279752492904663
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3544563949108124
current mse: 0.26944059133529663
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25169482827186584
current mse: 0.267652302980423
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26942506432533264
current mse: 0.26756900548934937
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27309608459472656
current mse: 0.27118980884552
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34900644421577454
current mse: 0.2703389823436737
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2791233956813812
current mse: 0.2749954164028168
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30091559886932373
current mse: 0.26874297857284546
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30158716440200806
current mse: 0.26968854665756226
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25747233629226685
current mse: 0.27672556042671204
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3095702826976776
current mse: 0.2814970314502716
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26808491349220276
current mse: 0.2739001512527466
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2821396589279175
current mse: 0.27362364530563354
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2747005224227905
current mse: 0.2717786729335785
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32037532329559326
current mse: 0.26821041107177734
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35171112418174744
current mse: 0.2677794396877289
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2773198187351227
current mse: 0.27251997590065
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32339057326316833
current mse: 0.27836301922798157
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2842315137386322
current mse: 0.26996779441833496
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2723582684993744
current mse: 0.26609814167022705
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30520573258399963
current mse: 0.26588255167007446
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3702649772167206
current mse: 0.2727009952068329
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3160015344619751
current mse: 0.27288657426834106
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3357338309288025
current mse: 0.26664721965789795
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36354562640190125
current mse: 0.26452645659446716
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2823793590068817
current mse: 0.26665428280830383
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2694133222103119
current mse: 0.27723872661590576
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3930225968360901
current mse: 0.27519601583480835
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2809959053993225
current mse: 0.2800029218196869
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3067118227481842
current mse: 0.26753711700439453
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3012928068637848
current mse: 0.2682628333568573
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2659294903278351
current mse: 0.268187940120697
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36556386947631836
current mse: 0.2563380300998688
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30701518058776855
current mse: 0.2598850131034851
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28414493799209595
current mse: 0.2525879144668579
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35602661967277527
current mse: 0.2513669729232788
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30947989225387573
current mse: 0.26868271827697754
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35995054244995117
current mse: 0.26153603196144104
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.342538058757782
current mse: 0.2642991840839386
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38940373063087463
current mse: 0.2739501893520355
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.550625741481781
current mse: 0.28077375888824463
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3274610638618469
current mse: 0.29124584794044495
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29584091901779175
current mse: 0.2853255271911621
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.43732765316963196
current mse: 0.27718985080718994
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28220847249031067
current mse: 0.2779557704925537
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.375089555978775
current mse: 0.25938472151756287
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3195057809352875
current mse: 0.2652989625930786
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3931257426738739
current mse: 0.25747549533843994
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.318357914686203
current mse: 0.25701093673706055
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3128526210784912
current mse: 0.25804728269577026
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.286552369594574
current mse: 0.2597521245479584
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3366857171058655
current mse: 0.27437567710876465
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2986094057559967
current mse: 0.2858012020587921
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32433629035949707
current mse: 0.29574450850486755
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3060850501060486
current mse: 0.29610323905944824
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33069396018981934
current mse: 0.2990686893463135
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2623504102230072
current mse: 0.2809615135192871
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3261690139770508
current mse: 0.27190110087394714
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2998376488685608
current mse: 0.2707071602344513
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34031087160110474
current mse: 0.2717440128326416
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2896284759044647
current mse: 0.27872708439826965
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2800423204898834
current mse: 0.2884822487831116
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28743186593055725
current mse: 0.28778553009033203
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2541521191596985
current mse: 0.28989124298095703
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3571385145187378
current mse: 0.28795477747917175
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28668975830078125
current mse: 0.28845497965812683
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31628888845443726
current mse: 0.2891266942024231
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3097761273384094
current mse: 0.2924097180366516
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2979728877544403
current mse: 0.292439341545105
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2879158854484558
current mse: 0.2919250726699829
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3611065447330475
current mse: 0.2907252311706543
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2896665930747986
current mse: 0.28530582785606384
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.40327611565589905
current mse: 0.28771504759788513
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25960680842399597
current mse: 0.29099953174591064
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29462793469429016
current mse: 0.2898714244365692
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.41698572039604187
current mse: 0.2904736399650574
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33033645153045654
current mse: 0.2709847390651703
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31837666034698486
current mse: 0.25623664259910583
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32591915130615234
current mse: 0.25725454092025757
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2807880938053131
current mse: 0.2525264620780945
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28930142521858215
current mse: 0.2445903867483139
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2670789659023285
current mse: 0.24729444086551666
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2675803601741791
current mse: 0.24807099997997284
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2805199921131134
current mse: 0.2535260319709778
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2688634693622589
current mse: 0.25718042254447937
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25092726945877075
current mse: 0.24845589697360992
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2667318284511566
current mse: 0.24366450309753418
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2936994731426239
current mse: 0.2594711184501648
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28252947330474854
current mse: 0.24646161496639252
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31722280383110046
current mse: 0.25310826301574707
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2763071358203888
current mse: 0.26285359263420105
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3219580054283142
current mse: 0.2664232850074768
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3028872311115265
current mse: 0.2614707052707672
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29807522892951965
current mse: 0.25985875725746155
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34838107228279114
current mse: 0.26399263739585876
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28255555033683777
current mse: 0.25044602155685425
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2974339723587036
current mse: 0.2524701654911041
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28635135293006897
current mse: 0.2541942000389099
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2937347888946533
current mse: 0.24271845817565918
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3750878870487213
current mse: 0.2332160472869873
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.259246826171875
current mse: 0.2326609194278717
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2696898579597473
current mse: 0.2302529513835907
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2672642767429352
current mse: 0.22687900066375732
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27548500895500183
current mse: 0.23140792548656464
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2898767292499542
current mse: 0.23635333776474
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2941088080406189
current mse: 0.2435436248779297
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31938737630844116
current mse: 0.24987977743148804
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2574450671672821
current mse: 0.2528747618198395
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3325860798358917
current mse: 0.25132858753204346
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2686437666416168
current mse: 0.2525063157081604
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3576450049877167
current mse: 0.24666671454906464
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.284594863653183
current mse: 0.24472764134407043
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28560319542884827
current mse: 0.24710269272327423
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4163420796394348
current mse: 0.2388063222169876
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35589098930358887
current mse: 0.24928739666938782
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26601603627204895
current mse: 0.2583903670310974
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32986992597579956
current mse: 0.24774880707263947
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29540812969207764
current mse: 0.2527182102203369
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29175081849098206
current mse: 0.25767722725868225
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3116685748100281
current mse: 0.2511883080005646
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3545197546482086
current mse: 0.24749118089675903
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.250397652387619
current mse: 0.2568496763706207
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31510066986083984
current mse: 0.25962862372398376
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31916189193725586
current mse: 0.24770677089691162
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3066716194152832
current mse: 0.2332611232995987
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2738974988460541
current mse: 0.22545650601387024
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2755262851715088
current mse: 0.22361227869987488
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2633509635925293
current mse: 0.23138555884361267
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.265726238489151
current mse: 0.24313940107822418
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24870963394641876
current mse: 0.24279473721981049
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2533522844314575
current mse: 0.24298164248466492
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2791402041912079
current mse: 0.24686098098754883
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27655160427093506
current mse: 0.24817262589931488
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2673172056674957
current mse: 0.2518247663974762
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2534943222999573
current mse: 0.23895736038684845
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3400251865386963
current mse: 0.22896040976047516
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3029736280441284
current mse: 0.21868406236171722
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2764028012752533
current mse: 0.21324540674686432
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.41811493039131165
current mse: 0.21246373653411865
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2962026000022888
current mse: 0.2187795490026474
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2843528687953949
current mse: 0.2107425034046173
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27744540572166443
current mse: 0.2080884873867035
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32475006580352783
current mse: 0.20982851088047028
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3175528645515442
current mse: 0.2088436782360077
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29510411620140076
current mse: 0.20717673003673553
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26148176193237305
current mse: 0.21020770072937012
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28923743963241577
current mse: 0.21844014525413513
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28861722350120544
current mse: 0.21126623451709747
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32082805037498474
current mse: 0.21787695586681366
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32433608174324036
current mse: 0.21706214547157288
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2743886709213257
current mse: 0.21671652793884277
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27334827184677124
current mse: 0.2261873185634613
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3083333969116211
current mse: 0.22514708340168
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.406476229429245
current mse: 0.22081121802330017
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28888797760009766
current mse: 0.22446087002754211
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3344869017601013
current mse: 0.22718966007232666
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2870446741580963
current mse: 0.22255779802799225
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.326284259557724
current mse: 0.21440522372722626
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25816115736961365
current mse: 0.20986910164356232
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26412081718444824
current mse: 0.20877021551132202
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3231416940689087
current mse: 0.20876142382621765
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27468106150627136
current mse: 0.2151016741991043
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29577136039733887
current mse: 0.22753240168094635
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3174642324447632
current mse: 0.22892071306705475
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3312002420425415
current mse: 0.21950271725654602
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39071938395500183
current mse: 0.21188576519489288
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31293076276779175
current mse: 0.21812906861305237
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3725961446762085
current mse: 0.21496374905109406
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2984960377216339
current mse: 0.2138901799917221
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2986048758029938
current mse: 0.21902930736541748
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29661113023757935
current mse: 0.21714557707309723
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.279724657535553
current mse: 0.2219710797071457
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30565059185028076
current mse: 0.2327205389738083
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3280330002307892
current mse: 0.23753921687602997
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2631562054157257
current mse: 0.24607789516448975
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3325408399105072
current mse: 0.2522306740283966
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2533569633960724
current mse: 0.24632136523723602
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29754018783569336
current mse: 0.24514201283454895
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2693096995353699
current mse: 0.24643942713737488
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2951364815235138
current mse: 0.2370663583278656
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35888999700546265
current mse: 0.23599445819854736
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2765458822250366
current mse: 0.2348945289850235
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2576303780078888
current mse: 0.2354598492383957
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31162700057029724
current mse: 0.2352045178413391
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3176755905151367
current mse: 0.22886691987514496
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36108988523483276
current mse: 0.22882303595542908
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2982172667980194
current mse: 0.22600777447223663
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36602097749710083
current mse: 0.2269330769777298
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.262603223323822
current mse: 0.2192230522632599
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34039273858070374
current mse: 0.22072552144527435
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31110894680023193
current mse: 0.22304849326610565
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28960806131362915
current mse: 0.2234354466199875
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2886432111263275
current mse: 0.22381603717803955
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35298627614974976
current mse: 0.23347976803779602
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29162654280662537
current mse: 0.23489174246788025
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30391550064086914
current mse: 0.23834705352783203
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27650949358940125
current mse: 0.25168082118034363
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2790345847606659
current mse: 0.26703977584838867
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25035083293914795
current mse: 0.2704159617424011
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2832725942134857
current mse: 0.24636946618556976
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25244173407554626
current mse: 0.25157907605171204
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2599872946739197
current mse: 0.2530936300754547
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2689591944217682
current mse: 0.24611371755599976
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3045847713947296
current mse: 0.257916122674942
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27905067801475525
current mse: 0.24958795309066772
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29397451877593994
current mse: 0.25689974427223206
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3854048550128937
current mse: 0.254534512758255
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39287930727005005
current mse: 0.25730618834495544
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28200069069862366
current mse: 0.2615607976913452
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34845009446144104
current mse: 0.25972914695739746
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2679368555545807
current mse: 0.2503344416618347
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27365267276763916
current mse: 0.25858816504478455
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32145020365715027
current mse: 0.2571450173854828
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26501330733299255
current mse: 0.2476235032081604
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36039817333221436
current mse: 0.2402230203151703
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2667624056339264
current mse: 0.24134768545627594
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27364593744277954
current mse: 0.2443663626909256
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2824579179286957
current mse: 0.24858854711055756
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28050997853279114
current mse: 0.248792365193367
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2891586124897003
current mse: 0.2508181929588318
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33554694056510925
current mse: 0.25337672233581543
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27947741746902466
current mse: 0.2710699141025543
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24326011538505554
current mse: 0.269166499376297
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2836301922798157
current mse: 0.2716009318828583
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25609639286994934
current mse: 0.26093852519989014
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25122785568237305
current mse: 0.26354286074638367
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.289774626493454
current mse: 0.26666396856307983
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26450294256210327
current mse: 0.2642368972301483
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3094891309738159
current mse: 0.2687356173992157
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28304198384284973
current mse: 0.27977579832077026
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26651501655578613
current mse: 0.2897736430168152
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3096861243247986
current mse: 0.29386845231056213
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2797313630580902
current mse: 0.2924933433532715
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29304197430610657
current mse: 0.29037347435951233
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2688172161579132
current mse: 0.2920154929161072
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31265538930892944
current mse: 0.29222580790519714
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24344737827777863
current mse: 0.296876460313797
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26971009373664856
current mse: 0.30270981788635254
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3178984522819519
current mse: 0.29260578751564026
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26670369505882263
current mse: 0.2913353741168976
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24977552890777588
current mse: 0.28699991106987
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3131661117076874
current mse: 0.29585421085357666
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2961612641811371
current mse: 0.28888681530952454
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2977422773838043
current mse: 0.281820148229599
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26221439242362976
current mse: 0.281907856464386
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2813461124897003
current mse: 0.2782302498817444
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27855411171913147
current mse: 0.27378931641578674
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3097250759601593
current mse: 0.2741570472717285
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2852720022201538
current mse: 0.28238487243652344
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32119569182395935
current mse: 0.2877103090286255
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2910459041595459
current mse: 0.28768324851989746
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26496702432632446
current mse: 0.28398996591567993
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24554862082004547
current mse: 0.29303205013275146
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3286099433898926
current mse: 0.2976079285144806
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2643302381038666
current mse: 0.3056846857070923
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26844266057014465
current mse: 0.3125474452972412
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3350866436958313
current mse: 0.3114102780818939
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3526897430419922
current mse: 0.31012535095214844
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.295531302690506
current mse: 0.31386303901672363
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.276411771774292
current mse: 0.3092455565929413
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25287389755249023
current mse: 0.3097010552883148
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3113730251789093
current mse: 0.313996285200119
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29686209559440613
current mse: 0.3111872375011444
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31104910373687744
current mse: 0.309680312871933
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27460694313049316
current mse: 0.3121199905872345
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30705106258392334
current mse: 0.30807411670684814
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26007479429244995
current mse: 0.3030169904232025
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2695673406124115
current mse: 0.2878110110759735
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25563305616378784
current mse: 0.2686708867549896
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30800196528434753
current mse: 0.261721670627594
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27990493178367615
current mse: 0.2663792371749878
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30675268173217773
current mse: 0.27911385893821716
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2961573302745819
current mse: 0.29076841473579407
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3151879608631134
current mse: 0.301632285118103
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2556755244731903
current mse: 0.32121527194976807
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25189507007598877
current mse: 0.3191277086734772
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28726905584335327
current mse: 0.31577372550964355
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25385957956314087
current mse: 0.3088061213493347
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.23873327672481537
current mse: 0.31476330757141113
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2771890461444855
current mse: 0.3210704028606415
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26507237553596497
current mse: 0.31947800517082214
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2687220573425293
current mse: 0.316464364528656
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24873685836791992
current mse: 0.3141014277935028
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34704428911209106
current mse: 0.3121596574783325
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3143559992313385
current mse: 0.3122808337211609
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3088802099227905
current mse: 0.31314295530319214
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31831836700439453
current mse: 0.3078829348087311
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30543580651283264
current mse: 0.3073807656764984
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2677556872367859
current mse: 0.30899059772491455
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.37723106145858765
current mse: 0.3037394881248474
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2690647542476654
current mse: 0.30760228633880615
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.23540286719799042
current mse: 0.3029669523239136
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28536784648895264
current mse: 0.29438531398773193
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24720190465450287
current mse: 0.28529608249664307
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2767043709754944
current mse: 0.2955975830554962
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3188643753528595
current mse: 0.3159492015838623
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3437686562538147
current mse: 0.3409423530101776
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2631405293941498
current mse: 0.37663620710372925
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2956034541130066
current mse: 0.39868655800819397
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2927572727203369
current mse: 0.4053570032119751
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2566877603530884
current mse: 0.41747725009918213
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28305870294570923
current mse: 0.4242248237133026
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2566547691822052
current mse: 0.430096298456192
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2974642515182495
current mse: 0.42392611503601074
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28590670228004456
current mse: 0.4253215789794922
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36781588196754456
current mse: 0.4252031743526459
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2686920464038849
current mse: 0.42354366183280945
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3031933307647705
current mse: 0.4310426414012909
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28506675362586975
current mse: 0.43657565116882324
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29533350467681885
current mse: 0.43443208932876587
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2941683530807495
current mse: 0.4237513840198517
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2627989649772644
current mse: 0.42135536670684814
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3069641888141632
current mse: 0.4194374084472656
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3188086152076721
current mse: 0.42871755361557007
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3147395849227905
current mse: 0.4595169425010681
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26434460282325745
current mse: 0.5139290690422058
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24007000029087067
current mse: 0.5678485035896301
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30815237760543823
current mse: 0.5616254806518555
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24649159610271454
current mse: 0.5836873650550842
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3007272481918335
current mse: 0.5658288598060608
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2735493779182434
current mse: 0.539382815361023
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25952625274658203
current mse: 0.5479645729064941
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28173136711120605
current mse: 0.5505377650260925
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.23327331244945526
current mse: 0.5692123770713806
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3401249945163727
current mse: 0.593607485294342
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26267462968826294
current mse: 0.5916186571121216
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2655520737171173
current mse: 0.6267719268798828
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2981693744659424
current mse: 0.6230401396751404
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31764230132102966
current mse: 0.6112860441207886
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24526578187942505
current mse: 0.6112857460975647
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3113618791103363
current mse: 0.5957629680633545
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24984100461006165
current mse: 0.6129367351531982
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3646782636642456
current mse: 0.6136500239372253
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28093811869621277
current mse: 0.5987464785575867
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3082033693790436
current mse: 0.6008292436599731
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3098224103450775
current mse: 0.6196764707565308
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28585362434387207
current mse: 0.6314752101898193
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2808755040168762
current mse: 0.6315539479255676
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26724347472190857
current mse: 0.6242562532424927
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26057305932044983
current mse: 0.5971863865852356
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24459198117256165
current mse: 0.6237558722496033
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3035672605037689
current mse: 0.6835319399833679
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26452335715293884
current mse: 0.6764252185821533
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2979026734828949
current mse: 0.6425051689147949
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3389289379119873
current mse: 0.6373457312583923
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3237209618091583
current mse: 0.646304190158844
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2709784209728241
current mse: 0.5876901745796204
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.351484477519989
current mse: 0.5936071276664734
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24206705391407013
current mse: 0.561385452747345
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.23949052393436432
current mse: 0.5763680934906006
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.273432195186615
current mse: 0.5843372941017151
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29181957244873047
current mse: 0.5524021983146667
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30964377522468567
current mse: 0.5668070912361145
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27221786975860596
current mse: 0.606011688709259
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2830904722213745
current mse: 0.5823999643325806
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27696773409843445
current mse: 0.5549939274787903
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34843048453330994
current mse: 0.5489142537117004
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25102558732032776
current mse: 0.5303439497947693
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26270702481269836
current mse: 0.550666332244873
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3192909061908722
current mse: 0.5645130276679993
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2887856662273407
current mse: 0.5493654012680054
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2788439989089966
current mse: 0.552942156791687
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2985171675682068
current mse: 0.5577675104141235
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2615860402584076
current mse: 0.5883391499519348
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2644287347793579
current mse: 0.6186997890472412
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.23920227587223053
current mse: 0.6059074401855469
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2320127934217453
current mse: 0.5981999635696411
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.259380042552948
current mse: 0.5760000348091125
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25168490409851074
current mse: 0.5633355379104614
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31668469309806824
current mse: 0.5557937622070312
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2737581133842468
current mse: 0.5272456407546997
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32808929681777954
current mse: 0.4690106213092804
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3514384925365448
current mse: 0.4678535461425781
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2882494032382965
current mse: 0.5059881806373596
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3744625747203827
current mse: 0.5614150166511536
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3502810001373291
current mse: 0.5393587350845337
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3059462308883667
current mse: 0.5203282237052917
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24363215267658234
current mse: 0.5270780920982361
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25125646591186523
current mse: 0.5067731738090515
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2510612905025482
current mse: 0.46520307660102844
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27555814385414124
current mse: 0.4761419892311096
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3408174216747284
current mse: 0.4630705714225769
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31282922625541687
current mse: 0.44796547293663025
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27312517166137695
current mse: 0.4858730733394623
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2901609241962433
current mse: 0.5229082703590393
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28868812322616577
current mse: 0.5201775431632996
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.46574193239212036
current mse: 0.4831722378730774
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2731591761112213
current mse: 0.48431605100631714
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29289767146110535
current mse: 0.49493345618247986
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2621161639690399
current mse: 0.5122687816619873
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28470706939697266
current mse: 0.41380807757377625
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.299013614654541
current mse: 0.4120418131351471
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35820403695106506
current mse: 0.4010156989097595
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38543304800987244
current mse: 0.3746871054172516
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30141305923461914
current mse: 0.34698575735092163
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31933358311653137
current mse: 0.3731633722782135
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2904658913612366
current mse: 0.4037856459617615
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2815949022769928
current mse: 0.4024460017681122
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29855117201805115
current mse: 0.40489229559898376
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3139782249927521
current mse: 0.39979860186576843
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3106134831905365
current mse: 0.36105507612228394
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3219005763530731
current mse: 0.36907893419265747
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3102124333381653
current mse: 0.3706479072570801
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28065258264541626
current mse: 0.3943956196308136
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3439939320087433
current mse: 0.39594048261642456
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2796595096588135
current mse: 0.41032353043556213
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4024609327316284
current mse: 0.4098987579345703
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3244314193725586
current mse: 0.4041518568992615
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3403126895427704
current mse: 0.36800116300582886
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.428840696811676
current mse: 0.35500311851501465
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3270261287689209
current mse: 0.3542070686817169
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2757236659526825
current mse: 0.35129010677337646
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3433246612548828
current mse: 0.3672630488872528
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34759435057640076
current mse: 0.3795703053474426
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30270320177078247
current mse: 0.3538804054260254
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29537296295166016
current mse: 0.3452640771865845
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3325200378894806
current mse: 0.3355899453163147
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3168295621871948
current mse: 0.33994075655937195
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3056272268295288
current mse: 0.341272234916687
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4206981956958771
current mse: 0.3406749367713928
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.37146303057670593
current mse: 0.33981430530548096
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30371326208114624
current mse: 0.3449517786502838
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36469992995262146
current mse: 0.35980623960494995
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32227662205696106
current mse: 0.379014790058136
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32819023728370667
current mse: 0.3797838091850281
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30172839760780334
current mse: 0.37378570437431335
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33903029561042786
current mse: 0.3682561218738556
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33737704157829285
current mse: 0.39549410343170166
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3801218569278717
current mse: 0.4114416241645813
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28693443536758423
current mse: 0.414655864238739
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3848397135734558
current mse: 0.42500928044319153
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4036135673522949
current mse: 0.4162370264530182
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3883185386657715
current mse: 0.4184531271457672
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32116517424583435
current mse: 0.38777899742126465
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3637808859348297
current mse: 0.3719971477985382
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28850916028022766
current mse: 0.35551443696022034
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30964982509613037
current mse: 0.33569851517677307
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2621980309486389
current mse: 0.3205926716327667
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30994853377342224
current mse: 0.31736302375793457
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33359524607658386
current mse: 0.3155808448791504
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3371678292751312
current mse: 0.31388071179389954
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4301134943962097
current mse: 0.3259613811969757
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4233437478542328
current mse: 0.3258974254131317
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30192503333091736
current mse: 0.3199428915977478
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3187953531742096
current mse: 0.31653377413749695
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.419339120388031
current mse: 0.30227503180503845
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4293566644191742
current mse: 0.3046610653400421
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30459779500961304
current mse: 0.31291064620018005
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3614419400691986
current mse: 0.31723785400390625
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.283596009016037
current mse: 0.3167913556098938
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2889973819255829
current mse: 0.3043939471244812
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.322966992855072
current mse: 0.31590521335601807
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31090423464775085
current mse: 0.3197999894618988
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3874712586402893
current mse: 0.3177809715270996
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3807225525379181
current mse: 0.31370851397514343
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33207279443740845
current mse: 0.31309977173805237
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31969118118286133
current mse: 0.32462283968925476
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36397504806518555
current mse: 0.3069995045661926
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27198609709739685
current mse: 0.2956896126270294
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3103019595146179
current mse: 0.2829776406288147
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3690241873264313
current mse: 0.2733457386493683
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3138124346733093
current mse: 0.2687411904335022
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27571791410446167
current mse: 0.2746988534927368
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2694709599018097
current mse: 0.27631494402885437
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34086716175079346
current mse: 0.2922936677932739
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28340011835098267
current mse: 0.2998645603656769
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27914899587631226
current mse: 0.31678420305252075
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3426058292388916
current mse: 0.31897270679473877
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2894349992275238
current mse: 0.32285618782043457
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2734425365924835
current mse: 0.3232753574848175
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3283151388168335
current mse: 0.31096625328063965
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3699575960636139
current mse: 0.304097056388855
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3195550739765167
current mse: 0.3049059212207794
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34549540281295776
current mse: 0.3152172267436981
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31567010283470154
current mse: 0.3217608332633972
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35847917199134827
current mse: 0.32659485936164856
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3969075381755829
current mse: 0.31989461183547974
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2998870611190796
current mse: 0.3285260498523712
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3252202868461609
current mse: 0.32219427824020386
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31073054671287537
current mse: 0.3190996050834656
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36027270555496216
current mse: 0.31857964396476746
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3186793923377991
current mse: 0.3128418028354645
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2968062162399292
current mse: 0.3074817657470703
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28117620944976807
current mse: 0.3107168972492218
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31068187952041626
current mse: 0.3245985507965088
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3145766854286194
current mse: 0.30827081203460693
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3188764154911041
current mse: 0.2987712323665619
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3649646043777466
current mse: 0.30179646611213684
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3914531171321869
current mse: 0.31734752655029297
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38920074701309204
current mse: 0.3073103427886963
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3716106116771698
current mse: 0.3119262456893921
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.323179692029953
current mse: 0.29442042112350464
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28256547451019287
current mse: 0.3105894923210144
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4012107849121094
current mse: 0.2940335273742676
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38972654938697815
current mse: 0.2951368987560272
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31139639019966125
current mse: 0.264232337474823
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4345976412296295
current mse: 0.2701214849948883
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34114351868629456
current mse: 0.27828100323677063
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3765891492366791
current mse: 0.29434940218925476
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3415868878364563
current mse: 0.2911430895328522
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3627188801765442
current mse: 0.2869459390640259
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32467010617256165
current mse: 0.2939469516277313
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.44902339577674866
current mse: 0.2893359959125519
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3364866375923157
current mse: 0.2807685136795044
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3947390913963318
current mse: 0.27497488260269165
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3529764413833618
current mse: 0.27760937809944153
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3465006649494171
current mse: 0.282094806432724
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.40602293610572815
current mse: 0.2831326127052307
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29194772243499756
current mse: 0.29092636704444885
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.38296401500701904
current mse: 0.2836476266384125
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3199653625488281
current mse: 0.2870267331600189
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3552058935165405
current mse: 0.2973305284976959
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39284199476242065
current mse: 0.2865346372127533
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34506046772003174
current mse: 0.2870841324329376
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31875309348106384
current mse: 0.2801028788089752
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4188772439956665
current mse: 0.2803468108177185
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31352394819259644
current mse: 0.2918550670146942
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35917314887046814
current mse: 0.3040347993373871
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39300307631492615
current mse: 0.2785555422306061
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34714293479919434
current mse: 0.255903035402298
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34048280119895935
current mse: 0.24778427183628082
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.288912296295166
current mse: 0.24873483180999756
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35158416628837585
current mse: 0.2603265941143036
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3169131278991699
current mse: 0.25788626074790955
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30946311354637146
current mse: 0.2588268220424652
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34900638461112976
current mse: 0.26338377594947815
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3292815685272217
current mse: 0.25588902831077576
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39294272661209106
current mse: 0.25932979583740234
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35035741329193115
current mse: 0.26739501953125
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35671043395996094
current mse: 0.25661852955818176
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32511162757873535
current mse: 0.2594376504421234
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3652489185333252
current mse: 0.2620874047279358
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36009839177131653
current mse: 0.28485509753227234
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3164595663547516
current mse: 0.3067935109138489
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3431112468242645
current mse: 0.30230990052223206
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29451900720596313
current mse: 0.3094651401042938
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.37841033935546875
current mse: 0.3124885857105255
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31857335567474365
current mse: 0.3214852809906006
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3998854458332062
current mse: 0.33608514070510864
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3230153024196625
current mse: 0.3340449929237366
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33668598532676697
current mse: 0.3135325312614441
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34495213627815247
current mse: 0.31737038493156433
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3971565067768097
current mse: 0.31031230092048645
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3026251196861267
current mse: 0.3159630596637726
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3483716547489166
current mse: 0.3186187744140625
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4128964841365814
current mse: 0.32147207856178284
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35810089111328125
current mse: 0.3324580490589142
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4131717383861542
current mse: 0.32237330079078674
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36543047428131104
current mse: 0.3142482042312622
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4039972424507141
current mse: 0.3062450587749481
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3310997188091278
current mse: 0.29676562547683716
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4412733018398285
current mse: 0.31494930386543274
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4340132474899292
current mse: 0.30247437953948975
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3417554199695587
current mse: 0.29312291741371155
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3573518395423889
current mse: 0.31863245368003845
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3597806692123413
current mse: 0.34301817417144775
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3192356526851654
current mse: 0.3387584686279297
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.5102226138114929
current mse: 0.33408254384994507
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.48093822598457336
current mse: 0.2984082102775574
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3673217296600342
current mse: 0.2846389710903168
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33020254969596863
current mse: 0.28122007846832275
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30378293991088867
current mse: 0.28022968769073486
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30270159244537354
current mse: 0.2837882339954376
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35437896847724915
current mse: 0.2801445722579956
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3148132264614105
current mse: 0.2852918803691864
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28719478845596313
current mse: 0.3014499247074127
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3149019181728363
current mse: 0.30583909153938293
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3535326421260834
current mse: 0.3091807961463928
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30981433391571045
current mse: 0.31174880266189575
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3278424143791199
current mse: 0.30185744166374207
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27404388785362244
current mse: 0.30443230271339417
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3522394895553589
current mse: 0.30034178495407104
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39432498812675476
current mse: 0.2985747456550598
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39092817902565
current mse: 0.29896318912506104
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4131266176700592
current mse: 0.2965409755706787
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31120210886001587
current mse: 0.29866474866867065
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3272397220134735
current mse: 0.30278560519218445
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34282517433166504
current mse: 0.30024486780166626
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2995659410953522
current mse: 0.2953264117240906
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3746448755264282
current mse: 0.29394811391830444
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.39343392848968506
current mse: 0.2886624336242676
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3410608768463135
current mse: 0.28326624631881714
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.326519250869751
current mse: 0.2794388234615326
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3036269247531891
current mse: 0.27198395133018494
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28757816553115845
current mse: 0.27779316902160645
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30387812852859497
current mse: 0.27777108550071716
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29636573791503906
current mse: 0.2795122563838959
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3411470353603363
current mse: 0.2789500057697296
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3261406719684601
current mse: 0.28591597080230713
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3240634500980377
current mse: 0.3021094501018524
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3378784954547882
current mse: 0.3069729506969452
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2912587821483612
current mse: 0.2916164994239807
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3362303078174591
current mse: 0.28997430205345154
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3357485234737396
current mse: 0.2870413064956665
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3416236937046051
current mse: 0.2903975248336792
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30475857853889465
current mse: 0.2903730869293213
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3176898658275604
current mse: 0.28717169165611267
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29194748401641846
current mse: 0.2887270450592041
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29184648394584656
current mse: 0.2892221510410309
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4036779999732971
current mse: 0.2920057773590088
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30762484669685364
current mse: 0.28717583417892456
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29302728176116943
current mse: 0.2883354723453522
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31016620993614197
current mse: 0.281708687543869
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2856588661670685
current mse: 0.27688172459602356
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25670045614242554
current mse: 0.26272037625312805
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33659255504608154
current mse: 0.25759536027908325
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28242865204811096
current mse: 0.2629392743110657
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2840811014175415
current mse: 0.25982406735420227
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3286622166633606
current mse: 0.2694123387336731
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3138211965560913
current mse: 0.27396848797798157
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3039868474006653
current mse: 0.2737424671649933
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3045613169670105
current mse: 0.268589586019516
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29727408289909363
current mse: 0.2739677429199219
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29288458824157715
current mse: 0.2735302448272705
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31683826446533203
current mse: 0.26954877376556396
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31196796894073486
current mse: 0.268391489982605
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34198838472366333
current mse: 0.2760245203971863
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2678740620613098
current mse: 0.2693175673484802
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28580373525619507
current mse: 0.2730378210544586
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32240620255470276
current mse: 0.2808617651462555
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3201111853122711
current mse: 0.2773803770542145
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3492763042449951
current mse: 0.28546565771102905
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34506312012672424
current mse: 0.28744250535964966
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30228039622306824
current mse: 0.29308104515075684
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27657178044319153
current mse: 0.2827514410018921
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3411198854446411
current mse: 0.274230033159256
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3334178030490875
current mse: 0.2785622775554657
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2575385272502899
current mse: 0.27254071831703186
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2824532389640808
current mse: 0.25557345151901245
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25978389382362366
current mse: 0.25970861315727234
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2849633991718292
current mse: 0.2489430010318756
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2870936393737793
current mse: 0.2527366280555725
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34477511048316956
current mse: 0.2647431492805481
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27253490686416626
current mse: 0.2732957601547241
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3472576439380646
current mse: 0.27985304594039917
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30519649386405945
current mse: 0.2776200473308563
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28196045756340027
current mse: 0.27254629135131836
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3340963125228882
current mse: 0.27051353454589844
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26495760679244995
current mse: 0.26744112372398376
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.312427282333374
current mse: 0.26720142364501953
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32081469893455505
current mse: 0.2760339379310608
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.33693087100982666
current mse: 0.26734641194343567
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3028724789619446
current mse: 0.268879771232605
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2880777418613434
current mse: 0.27551642060279846
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27141186594963074
current mse: 0.27554693818092346
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4119325280189514
current mse: 0.27834251523017883
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.4138049781322479
current mse: 0.2842067778110504
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3248516917228699
current mse: 0.277682900428772
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3439311683177948
current mse: 0.27690213918685913
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3175143599510193
current mse: 0.2783539891242981
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3207268714904785
current mse: 0.27942991256713867
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2888583242893219
current mse: 0.2717796266078949
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29306384921073914
current mse: 0.2724500298500061
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27603045105934143
current mse: 0.277798056602478
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2664158046245575
current mse: 0.2741663455963135
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29683756828308105
current mse: 0.2781519293785095
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3408295810222626
current mse: 0.2900330424308777
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29360905289649963
current mse: 0.2900714576244354
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.294172078371048
current mse: 0.28789424896240234
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.35427120327949524
current mse: 0.28693854808807373
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2608996629714966
current mse: 0.29496684670448303
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2967020273208618
current mse: 0.30122941732406616
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3209543824195862
current mse: 0.30419373512268066
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27492520213127136
current mse: 0.3023633360862732
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3622942864894867
current mse: 0.3003512918949127
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29746130108833313
current mse: 0.29761144518852234
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29935353994369507
current mse: 0.2909661531448364
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36099717020988464
current mse: 0.2937600612640381
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2772800028324127
current mse: 0.29912814497947693
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2798685133457184
current mse: 0.29545658826828003
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29638388752937317
current mse: 0.28925466537475586
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31930699944496155
current mse: 0.2892352044582367
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3058087229728699
current mse: 0.294942706823349
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32559946179389954
current mse: 0.2984228730201721
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2850971221923828
current mse: 0.29178208112716675
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28467780351638794
current mse: 0.290569543838501
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27592727541923523
current mse: 0.29448646306991577
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3055483102798462
current mse: 0.30327221751213074
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27829912304878235
current mse: 0.3179568648338318
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3176140785217285
current mse: 0.31126585602760315
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3232755959033966
current mse: 0.31907492876052856
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3675316274166107
current mse: 0.3193361163139343
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3142717182636261
current mse: 0.32267701625823975
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2734644114971161
current mse: 0.3147284686565399
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26728951930999756
current mse: 0.3200933337211609
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2976459562778473
current mse: 0.3299369215965271
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.32371652126312256
current mse: 0.3490554988384247
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2975350320339203
current mse: 0.3719557821750641
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3458121418952942
current mse: 0.3659863770008087
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2942765951156616
current mse: 0.3510465621948242
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28168225288391113
current mse: 0.3230018615722656
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2984094023704529
current mse: 0.3151664435863495
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2713325023651123
current mse: 0.3078640103340149
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31776541471481323
current mse: 0.30725693702697754
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3541816771030426
current mse: 0.31089261174201965
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30188465118408203
current mse: 0.3059878349304199
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.34351372718811035
current mse: 0.3235868811607361
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.325165718793869
current mse: 0.3193097412586212
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3451535701751709
current mse: 0.3265722990036011
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.25805050134658813
current mse: 0.34051594138145447
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24005725979804993
current mse: 0.3486127555370331
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2960865795612335
current mse: 0.34349414706230164
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3121474087238312
current mse: 0.3396765887737274
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.27569082379341125
current mse: 0.327065110206604
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29151859879493713
current mse: 0.3318847417831421
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2881500720977783
current mse: 0.3268575370311737
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2625974118709564
current mse: 0.3232736587524414
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2730628252029419
current mse: 0.32484856247901917
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.24737422168254852
current mse: 0.33863988518714905
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3395547866821289
current mse: 0.34430578351020813
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28229764103889465
current mse: 0.33993104100227356
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.31510525941848755
current mse: 0.3730093836784363
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.42036497592926025
current mse: 0.36462724208831787
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.26897260546684265
current mse: 0.34637048840522766
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2771836221218109
current mse: 0.3371196985244751
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.30929845571517944
current mse: 0.3307928442955017
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29721376299858093
current mse: 0.3413531184196472
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2906661033630371
current mse: 0.3354617655277252
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29215675592422485
current mse: 0.3339519798755646
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28647884726524353
current mse: 0.3376583456993103
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3414524793624878
current mse: 0.3396233320236206
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3193972110748291
current mse: 0.343581885099411
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.29175686836242676
current mse: 0.3359210193157196
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3320285379886627
current mse: 0.3387034237384796
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.28334859013557434
current mse: 0.35338300466537476
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2890985608100891
current mse: 0.3549943268299103
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.2833704650402069
current mse: 0.3560958504676819
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.3724828064441681
current mse: 0.37118712067604065
ttt big iter 0 small iter 0, recon loss / batch_x.std 0.36271971464157104
current mse: 0.33961382508277893
test shape: (1001, 1, 96, 862) (1001, 1, 96, 862)
test shape: (1001, 96, 862) (1001, 96, 862)
mse:0.3138798177242279, mae:0.22324834764003754

'''

recon_loss = []
mse = []

# Define the regular expression pattern
pattern_recon_loss = re.compile(r'recon loss / batch_x\.std (\d+\.\d+)')
pattern_mse = re.compile(r'current mse: (\d+\.\d+)')

# Find all matches for recon loss and mse
matches_recon_loss = pattern_recon_loss.findall(log_text)
matches_mse = pattern_mse.findall(log_text)

# Convert the matches to float and add to lists
recon_loss = [float(match) for match in matches_recon_loss]
mse = [float(match) for match in matches_mse]

from matplotlib import pyplot as plt
# plot scatter plot for recon_loss and mse.
plt.scatter(recon_loss, mse)
plt.xlabel('recon_loss')
plt.ylabel('mse')
plt.title('recon_loss vs mse')
plt.savefig('recon_loss_vs_mse.png')