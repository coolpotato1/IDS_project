import matplotlib.pyplot as plt
import numpy as np
names = ["KDD_filtered", "MitM", "UDP_DOS", "SF"]
# accuracies_nn = [0.77, 0.85, 0.72, 0.9]
# accuracies_rf = [0.66, 0.55, 0.87, 0.54]
KDD_TEST_LENGTH = 19592
MITM_TEST_LENGTH = 1542
UDP_DOS_TEST_LENGTH = 12747
SVELTE_TEST_LENGTH = 2718
COMBINED_TEST_LENGTH = 36599
# The below results were for the individual datasets where each dataset was trained and evaluated on 4 times. This was the basis for dataset_complexity.png
# Precisions_nn = [[0.9664978125719549, 0.9653855059041445, 0.9648235429359696, 0.9659667743424089, 0.9655771905424201], [0.9453416149068323, 0.9532828282828283, 0.9453416149068323, 0.9498117942283564, 0.942998760842627], [0.9623115577889447, 0.9623115577889447, 0.9623115577889447, 0.9623115577889447, 0.9623115577889447], [0.9977477477477478, 0.9977477477477478, 0.9977477477477478, 0.9977477477477478, 0.9977477477477478]]
Recalls_nn = [[0.8496103633235502, 0.8439429207570084, 0.8494079546604595, 0.8473838680295517, 0.8431332861046453], [0.9465174129353234, 0.9390547263681592, 0.9465174129353234, 0.9415422885572139, 0.9465174129353234], [0.9960988296488946, 0.9960988296488946, 0.9960988296488946, 0.9960988296488946, 0.9960988296488946], [1.0, 1.0, 1.0, 1.0, 1.0]]
# Fscores_nn = [[0.9042925620724942, 0.9005885846967978, 0.9034445640473627, 0.9027979945010512, 0.9002107083040684], [0.9459291485394655, 0.9461152882205514, 0.9459291485394655, 0.9456589631480325, 0.9447548106765984], [0.9789137380191693, 0.9789137380191693, 0.9789137380191693, 0.9789137380191693, 0.9789137380191693], [0.9988726042841037, 0.9988726042841037, 0.9988726042841037, 0.9988726042841037, 0.9988726042841037]]
Accuracies_nn = [[0.9092997141690486, 0.9060330747243773, 0.9084320130665577, 0.9079726418946509, 0.9057268272764394], [0.943579766536965, 0.9442282749675746, 0.943579766536965, 0.943579766536965, 0.9422827496757458], [0.9974111555660156, 0.9974111555660156, 0.9974111555660156, 0.9974111555660156, 0.9974111555660156], [0.9992641648270787, 0.9992641648270787, 0.9992641648270787, 0.9992641648270787, 0.9992641648270787]]
#
# Precisions_rf = [[0.9698373114831967, 0.9698337292161521, 0.9686293436293436, 0.9696317551410808, 0.9697257509200997], [0.978021978021978, 0.9792176039119804, 0.9768009768009768, 0.9792176039119804, 0.9768292682926829], [0.8428571428571429, 0.6742756804214223, 0.6736842105263158, 0.6742756804214223, 0.6742756804214223], [0.9977477477477478, 0.9977477477477478, 0.9977477477477478, 0.9977477477477478, 0.9977477477477478]]
# Recalls_rf = [[0.8265357757312013, 0.8264345713996559, 0.812468373646392, 0.8207671288331141, 0.8266369800627467], [0.996268656716418, 0.996268656716418, 0.9950248756218906, 0.996268656716418, 0.996268656716418], [0.9973992197659298, 0.9986996098829649, 0.9986996098829649, 0.9986996098829649, 0.9986996098829649], [1.0, 1.0, 1.0, 1.0, 1.0]]
# Fscores_rf = [[0.8924707682220523, 0.8924102508059669, 0.8837030106224888, 0.8890106878596876, 0.8924825174825175], [0.9870609981515712, 0.9876695437731196, 0.9858287122612446, 0.9876695437731196, 0.9864532019704433], [0.9136390708755212, 0.8050314465408804, 0.8046097433211105, 0.8050314465408804, 0.8050314465408804], [0.9988726042841037, 0.9988726042841037, 0.9988726042841037, 0.9988726042841037, 0.9988726042841037]]
# Accuracies_rf = [[0.8995508370763577, 0.8994997958350347, 0.8921498570845243, 0.8966414863209473, 0.8995508370763577], [0.9863813229571985, 0.9870298313878081, 0.9850843060959793, 0.9870298313878081, 0.9857328145265889], [0.9886247744567349, 0.9708166627441751, 0.9707382129128422, 0.9708166627441751, 0.9708166627441751], [0.9992641648270787, 0.9992641648270787, 0.9992641648270787, 0.9992641648270787, 0.9992641648270787]]


# The below results were on the combined dataset, where each classifier was run 10 times
Precisions_rf = [[0.9473684210526315, 0.9461531523405791, 0.9224764468371467, 0.9452852153667055, 0.9421317064417728, 0.9236614242799082, 0.9445218845832172, 0.9468066020810908, 0.9244058581075155, 0.9246821569487067]]
Recalls_rf = [[0.8591572123176662, 0.8500810372771475, 0.8331442463533225, 0.8554294975688816, 0.8509724473257698, 0.8471636952998379, 0.8236628849270664, 0.8553484602917342, 0.8542139384116694, 0.8546191247974068]]
Fscores_rf = [[0.9011091751306787, 0.8955478721133735, 0.8755375771769214, 0.8981154549708598, 0.8942348633228305, 0.8837602502324795, 0.8799619064109778, 0.8987568119891007, 0.8879248620646086, 0.8882712149926301]]
Accuracies_rf = [[0.9364190278422908, 0.9331402497335993, 0.9201344299024563, 0.934561053580699, 0.9321292931500861, 0.9248613350091532, 0.9242329025383207, 0.9350255471460969, 0.9272930954397661, 0.9275116806470122]]

# Batch size of 40
# Precisions_nn = [[0.9327225130890052, 0.9450192914766748, 0.9395157765929552, 0.9391447368421053, 0.9452472144846796, 0.8717054577893115, 0.9189794862954663, 0.9436243997865907, 0.8791091387245233, 0.8957214765100671]]
# Recalls_nn = [[0.8662074554294976, 0.8733387358184765, 0.8710696920583468, 0.8791734197730956, 0.8799837925445705, 0.8710696920583468, 0.8640194489465154, 0.859967585089141, 0.8668557536466774, 0.8652350081037277]]
# Fscores_nn = [[0.898235294117647, 0.9077661725067385, 0.9039989907909677, 0.9081700987778335, 0.9114487157965419, 0.8713874589599124, 0.8906524099908111, 0.8998558466887137, 0.8729394483433981, 0.8802143446001648]]
# Accuracies_nn = [[0.9338233285062434, 0.9401622995163802, 0.9376212464821443, 0.9400530069127572, 0.9423481515888412, 0.9133036421760158, 0.9284679909287139, 0.9354627175605891, 0.9149157080794558, 0.9205989234678543]]

# Batch size of 60
Precisions_nn = [[0.9372143165157395, 0.9463830535290498, 0.8694652098907418, 0.9437957476472638, 0.9391493055555555, 0.9431414971873647, 0.945080891998251, 0.9454338099833318, 0.9276123257014977, 0.9401716888577435]]
Recalls_nn = [[0.8806320907617504, 0.8725283630470017, 0.8576985413290114, 0.8777147487844409, 0.8767423014586709, 0.8831442463533226, 0.8757698541329011, 0.8733387358184765, 0.8733387358184765, 0.8697730956239871]]
Fscores_nn = [[0.9080426154167538, 0.9079563182527302, 0.863541794150043, 0.9095566006046356, 0.9068734283319362, 0.9121573550952082, 0.9091062039957939, 0.9079573697291378, 0.8996577343684782, 0.9036033002188921]]
Accuracies_nn = [[0.9398617448564168, 0.9403535615727205, 0.9086040602202246, 0.9411459329489876, 0.9392879586873958, 0.9426487062488046, 0.9409546708926474, 0.940298915270909, 0.9343151452225471, 0.9374299844258039]]

# Code for generating 4 bar plots, one for each evaluation metric
precision_nn = np.mean(Precisions_nn, axis=1)
recall_nn = np.mean(Recalls_nn, axis=1)
fscore_nn = np.mean(Fscores_nn, axis=1)
accuracy_nn = np.mean(Accuracies_nn, axis=1)


precision_rf = np.mean(Precisions_rf, axis=1)
recall_rf = np.mean(Recalls_rf, axis=1)
fscore_rf = np.mean(Fscores_rf, axis=1)
accuracy_rf = np.mean(Accuracies_rf, axis=1)

index = np.arange(len(precision_nn))
width = 0.3

plt.title("Test")
plt.subplot(411)
plt.bar(index - width, accuracy_nn, width=width, color='C1', align='center', label='DNN')
plt.bar(index, accuracy_rf, width=width, color='C2', align='center', label='RF')
plt.ylabel("Accuracy")
plt.xticks(index - width/2, names)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Dataset accuracies for RF and DNN")

plt.subplot(412)
plt.bar(index - width, recall_nn, width=width, color='C1', align='center', label='DNN')
plt.bar(index, recall_rf, width=width, color='C2', align='center', label='RF')
plt.ylabel("Recall")
plt.xticks(index - width/2, names)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Dataset recalls for RF and DNN")

plt.subplot(413)
plt.bar(index - width, precision_nn, width=width, color='C1', align='center', label='DNN')
plt.bar(index, precision_rf, width=width, color='C2', align='center', label='RF')
plt.ylabel("Precision")
plt.xticks(index - width/2, names)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("Precision on datasets for RF and DNN")

plt.subplot(414)
plt.bar(index - width, fscore_nn, width=width, color='C1', align='center', label='DNN')
plt.bar(index, fscore_rf, width=width, color='C2', align='center', label='RF')
plt.ylabel("F1-score")
plt.xticks(index - width/2, names)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title("F1-score on the datasets for RF and DNN")

plt.show()

# attack_type_recalls = [0.850402144772118, 0.8628665840561751, 0.9626865671641791, 0.9973992197659298, 1.0]
# index = np.arange(len(attack_type_recalls))
# width = 0.3
# plt.subplot(111)
# plt.bar(index, attack_type_recalls, width=width, color='C7', align='center')
# plt.ylabel("Recall")
# plt.ylim([0.5, 1.05])
# plt.xticks(index, ["KDD-DoS", "KDD-Probe", "MitM", "UDP-DoS", "SF"])
# plt.title("Recalls for the different attack types in the combined dataset")
# plt.show()

# probe_recalls = [0.9374149659863945, 0.9929078014184397, 1.0, 1.0]
# novel_probe_recalls =[0.7228915662650602, 0.9717868338557993]
#
# dos_recalls = [0.766016713091922, 1.0, 0.9995705389735882, 0.9512195121951219, 0.9909774436090225, 1.0]
# novel_dos_recalls = [0.7842605156037992, 0.5, 0.17226277372262774, 0.0, 0.0]
#
# index = np.arange(len(probe_recalls))
# width = 0.25
# plt.subplot(212)
# plt.bar(index, probe_recalls, width=width, color='C7', align='center', label='Attacks included in training')
# plt.bar([4, 5], novel_probe_recalls, width=width, color='C8', align='center', label='Novel attacks')
# plt.ylabel("Recall")
# plt.ylim([0, 1.05])
# plt.xticks(np.append(index, [4, 5]), ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"])
# plt.title("Recalls for the different KDD Probe attacks")
# plt.legend()
#
# index = np.arange(len(dos_recalls))
# index2 = np.arange(len(dos_recalls), len(dos_recalls) + len(novel_dos_recalls))
# plt.subplot(211)
# plt.bar(index, dos_recalls, width=width, color='C7', align='center', label='Attacks included in training')
# plt.bar(index2, novel_dos_recalls, width=width, color='C8', align='center', label='Novel attacks')
# plt.ylabel("Recall")
# plt.ylim([0, 1.05])
# plt.xticks(np.append(index, index2), ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
#                   "udpstorm", "processtable", "worm", "mailbomb"])
# plt.legend()
# plt.title("Recalls for the different KDD DoS attacks")
# plt.show()

# recall_and_fpr = [(1.0, 1.0), (0.8992706645056726, 0.31773774681561484), (0.8942463533225283, 0.12399521826950823), (0.8923014586709886, 0.0869368069582423), (0.8893841166936791, 0.07048930293911539), (0.8876012965964344, 0.05696854775547219), (0.8863047001620745, 0.0329774516674224), (0.8830632090761751, 0.029720928315264437), (0.8819286871961102, 0.027907168473556204), (0.880064829821718, 0.026711735850612144), (0.8772285251215559, 0.02514530689640958), (0.8751215559157213, 0.024609423306813965), (0.8733387358184765, 0.024073539717218352), (0.8715559157212318, 0.023743765200544126), (0.7974878444084279, 0.014180304216991632), (0.7956239870340357, 0.013850529700317409), (0.7917341977309562, 0.013438311554474627), (0.7872771474878444, 0.013026093408631848), (0.7804700162074554, 0.012448988004451956), (0.7715559157212317, 0.011871882600272063)]
# recalls = [element[0] for element in recall_and_fpr]
# fprs = [element[1] for element in recall_and_fpr]
# print(fprs[9])
# recalls.append(0)
# fprs.append(0)
# plt.subplot(111)
# plt.plot(fprs, recalls, color='C8')
# plt.plot(fprs[9], recalls[9], color='grey', marker='o')
# plt.ylim([0, 1.05])
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xticks(np.arange(0, 1.1, 0.1))
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate (Recall)")
# plt.title("ROC curve for the DNN on the combined dataset")
# plt.show()

# The recalls, accuracies and time it took to classify the 4 datasets when they were part of the combined dataset
# (with the combined dataset being included as the last value)
# recall_combined = [0.8446513510778262, 0.9567164179104477, 0.998699609882965, 1.0]
# accuracy_combined = [0.9069722335647203, 0.9491569390402075, 0.9539970189064094, 0.9635761589403973]
# Average_time = [0.2531238555908203, 0.03510198593139648, 0.13962650299072266, 0.04625396728515625, 0.3624127864837646]
# width = 0.3
# index = np.arange(len(recall_combined))
# plt.title("Test")
# plt.subplot(121)
# plt.bar(index - width, recall_nn, width=width, color='C7', align='center', label='Before')
# plt.bar(index, recall_combined, width=width, color='C8', align='center', label='After')
# plt.ylabel("Recall")
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xticks(index - width/2, names)
# plt.legend()
# plt.title("Recalls on the datasets")
# #
# plt.subplot(122)
# plt.bar(index - width, accuracy_nn, width=width, color='C7', align='center', label='Before')
# plt.bar(index, accuracy_combined, width=width, color='C8', align='center', label='After')
# plt.ylabel("Accuracy")
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xticks(index - width/2, names)
# plt.legend()
# plt.title("Accuracies on the datasets")
#
# plt.show()


# The time performance of the classifiers

nn_timing = [0.045476770401000975/SVELTE_TEST_LENGTH, 0.14261889457702637/UDP_DOS_TEST_LENGTH, 0.0350989818572998/MITM_TEST_LENGTH, 0.20045228004455568/KDD_TEST_LENGTH, 0.3925510883331299/COMBINED_TEST_LENGTH]
rf_timing = [0.03291583061218262/SVELTE_TEST_LENGTH, 0.12506561279296874/UDP_DOS_TEST_LENGTH, 0.038497400283813474/MITM_TEST_LENGTH, 0.29780354499816897/KDD_TEST_LENGTH, 0.5932202816009522/COMBINED_TEST_LENGTH]

index = np.arange(len(nn_timing))
plt.subplot(111)
plt.plot(index, [1000000 * element for element in nn_timing], color='C7', label='DNN')
plt.plot(index, [1000000 * element for element in rf_timing], color='C8', label='RF')
plt.xticks(index, ["SF", "UDP_DOS", "MitM", "KDD_filtered", "Combined Dataset"])
plt.ylabel("Prediction time per observation (μs)")
plt.ylim([0, 26])
plt.title("Prediction times for the RF and DNN classifiers")
plt.legend()

plt.show()

run1_prediction = [0.003122985999652883, 0.0017083120001188945, 0.001480433998949593, 0.0026836690012714826, 0.0025997890006692614, 0.0026386690005892888, 0.0026830690003407653, 0.0026612289984768722, 0.002514350002456922, 0.0016975129983620718, 0.0016601940005784854, 0.001933672003360698, 0.001683353999396786, 0.0016618730005575344, 0.0018512329988880083, 0.002202832001785282, 0.0023061510000843555, 0.0025940310006262735, 0.0025196310016326606, 0.002582270000857534]
run1_preprocessing = [0.003675699234008789, 0.001847982406616211, 0.0023717880249023438, 0.0024614334106445312, 0.001882791519165039, 0.0016837120056152344, 0.0019228458404541016, 0.0019075870513916016, 0.0019273757934570312, 0.0018842220306396484, 0.0019350051879882812, 0.0019867420196533203, 0.0027599334716796875, 0.001867055892944336, 0.0016977787017822266, 0.0018794536590576172, 0.0025391578674316406, 0.001984834671020508, 0.002070188522338867, 0.00189208984375]

run2_prediction = [0.00330947100155754, 0.0026332729976275004, 0.0026144330004171934, 0.0016283950026263483, 0.002795751999656204, 0.0014709560018673074, 0.0026043530015158467, 0.0023932730000524316, 0.0018400750013825018, 0.0025889930002449546, 0.002647313001943985, 0.0026618329975462984, 0.002599433999421308, 0.002580593001766829, 0.0025621140011935495, 0.002554553997470066, 0.0023102340019249823, 0.0021463140001287684, 0.0026674729997466784, 0.00255419400127721]
run2_preprocessing = [0.006483316421508789, 0.003748655319213867, 0.0038576126098632812, 0.0035185813903808594, 0.003802061080932617, 0.0035355091094970703, 0.0037529468536376953, 0.0035774707794189453, 0.004059553146362305, 0.003815174102783203, 0.0038890838623046875, 0.0037279129028320312, 0.0036928653717041016, 0.0036628246307373047, 0.003778696060180664, 0.003727436065673828, 0.003916263580322266, 0.003489255905151367, 0.003802776336669922, 0.003686189651489258]

run3_prediction = [0.0031539619994873647, 0.0026019399992947, 0.002625459001137642, 0.0025960569982999004, 0.0026519790008023847, 0.002630618000694085, 0.0026315770010114647, 0.0025112119983532466, 0.002558133001002716, 0.002565451999544166, 0.002585013000498293, 0.0025746909996087197, 0.002585011003247928, 0.002583329998742556, 0.0025911300035659224, 0.0026018089993158355, 0.0025601669985917397, 0.00197466699682991, 0.0025626870010455605, 0.002623527998366626]
run3_preprocessing = [0.023493528366088867, 0.02319192886352539, 0.020836591720581055, 0.020839452743530273, 0.022331953048706055, 0.020830154418945312, 0.021812915802001953, 0.020441293716430664, 0.020074129104614258, 0.020537614822387695, 0.021170854568481445, 0.021033048629760742, 0.0211336612701416, 0.020969629287719727, 0.020697593688964844, 0.02161431312561035, 0.02069234848022461, 0.02079939842224121, 0.02097320556640625, 0.02123737335205078]

run4_prediction = [0.005789291000837693, 0.0026969409991579596, 0.001091424001060659, 0.0026262579995091073, 0.0026723390001279768, 0.00263213800280937, 0.0026549379981588572, 0.0025786160003917757, 0.002556414998252876, 0.002011003001825884, 0.0025764559977687895, 0.0013868700007151347, 0.0020546830019156914, 0.0025619339976401534, 0.0019894010001735296, 0.0023243280011229217, 0.0019329999995534308, 0.0025570120014890563, 0.0024796109973976854, 0.0014157890000205953]
run4_preprocessing = [0.2062666416168213, 0.21045851707458496, 0.2170562744140625, 0.21826505661010742, 0.22381114959716797, 0.2223968505859375, 0.21058225631713867, 0.21381688117980957, 0.2149655818939209, 0.21251630783081055, 0.21799254417419434, 0.21633362770080566, 0.20887088775634766, 0.22531557083129883, 0.21799230575561523, 0.20633602142333984, 0.21872425079345703, 0.2141575813293457, 0.2057962417602539, 0.20897221565246582]

# index = np.arange(0, 4)
# plt.subplot(111)
# prediction_y = [np.mean(run1_prediction), np.mean(run2_prediction), np.mean(run3_prediction), np.mean(run4_prediction)]
# preprocessing_y = [np.mean(run1_preprocessing), np.mean(run2_preprocessing), np.mean(run3_preprocessing), np.mean(run4_preprocessing)]
# print(np.mean(prediction_y))
# plt.plot(index, prediction_y, color='C7', label='Classify flow')
# plt.plot(index, preprocessing_y, color='C8', label='Generate flow')
# plt.errorbar(index, prediction_y, [np.std(run1_prediction), np.std(run2_prediction), np.std(run3_prediction), np.std(run4_prediction)], capsize=3, color='C7')
# plt.errorbar(index, preprocessing_y, [np.std(run1_preprocessing), np.std(run2_preprocessing), np.std(run3_preprocessing), np.std(run4_preprocessing)], capsize=3, color='C8')
# plt.xticks(index, ["Run 1 (0.2 pps)", "Run 2 (2 pps)", "Run 3 (20 pps)", "Run 4 (200 pps)"])
# plt.legend()
# plt.yscale('log')
# plt.ylabel("Time (seconds)")
# plt.yticks([0.001, 0.01, 0.1, 0.5], ["0.001", "0.01", "0.1", "0.5"])
# plt.title("The time taken to generate and classify each flow")
# plt.show()