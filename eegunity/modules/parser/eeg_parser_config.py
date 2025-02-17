STANDARD_EEG_CHANNELS = ['AFp10h', 'AFp11h', 'AFF10h', 'AFF11h', 'FAF10h', 'FAF11h', 'FFC10h', 'FFC11h', 'FCF10h',
                         'FCF11h', 'FCC10h', 'FCC11h', 'CFC10h', 'CFC11h', 'CCP10h', 'CCP11h', 'CPC10h', 'CPC11h',
                         'CPP10h', 'CPP11h', 'PCP10h', 'PCP11h', 'PPO10h', 'PPO11h', 'POP10h', 'POP11h', 'POO10h',
                         'POO11h', 'OPO10h', 'OPO11h', 'AFC10h', 'AFC11h', 'CAF10h', 'CAF11h', 'FFT10h', 'FFT11h',
                         'FTF10h', 'FTF11h', 'FTT10h', 'FTT11h', 'TFT10h', 'TFT11h', 'TTP10h', 'TTP11h', 'TPP10h',
                         'TPP11h', 'Fp10h', 'Fp11h', 'AFp1h', 'AFp2h', 'AFp3h', 'AFp4h', 'AFp5h', 'AFp6h', 'AFp7h',
                         'AFp8h', 'AFp9h', 'AFp10', 'AFp11', 'AF10h', 'AF11h', 'FC10h', 'FC11h', 'CP10h', 'CP11h',
                         'PO10h', 'PO11h', 'FT10h', 'FT11h', 'TP10h', 'TP11h', 'AFF1h', 'AFF2h', 'AFF3h', 'AFF4h',
                         'AFF5h', 'AFF6h', 'AFF7h', 'AFF8h', 'AFF9h', 'AFF10', 'AFF11', 'FAF1h', 'FAF2h', 'FAF3h',
                         'FAF4h', 'FAF5h', 'FAF6h', 'FAF7h', 'FAF8h', 'FAF9h', 'FAF10', 'FAF11', 'FFC1h', 'FFC2h',
                         'FFC3h', 'FFC4h', 'FFC5h', 'FFC6h', 'FFC7h', 'FFC8h', 'FFC9h', 'FFC10', 'FFC11', 'FCF1h',
                         'FCF2h', 'FCF3h', 'FCF4h', 'FCF5h', 'FCF6h', 'FCF7h', 'FCF8h', 'FCF9h', 'FCF10', 'FCF11',
                         'FCC1h', 'FCC2h', 'FCC3h', 'FCC4h', 'FCC5h', 'FCC6h', 'FCC7h', 'FCC8h', 'FCC9h', 'FCC10',
                         'FCC11', 'CFC1h', 'CFC2h', 'CFC3h', 'CFC4h', 'CFC5h', 'CFC6h', 'CFC7h', 'CFC8h', 'CFC9h',
                         'CFC10', 'CFC11', 'CCP1h', 'CCP2h', 'CCP3h', 'CCP4h', 'CCP5h', 'CCP6h', 'CCP7h', 'CCP8h',
                         'CCP9h', 'CCP10', 'CCP11', 'CPC1h', 'CPC2h', 'CPC3h', 'CPC4h', 'CPC5h', 'CPC6h', 'CPC7h',
                         'CPC8h', 'CPC9h', 'CPC10', 'CPC11', 'CPP1h', 'CPP2h', 'CPP3h', 'CPP4h', 'CPP5h', 'CPP6h',
                         'CPP7h', 'CPP8h', 'CPP9h', 'CPP10', 'CPP11', 'PCP1h', 'PCP2h', 'PCP3h', 'PCP4h', 'PCP5h',
                         'PCP6h', 'PCP7h', 'PCP8h', 'PCP9h', 'PCP10', 'PCP11', 'PPO1h', 'PPO2h', 'PPO3h', 'PPO4h',
                         'PPO5h', 'PPO6h', 'PPO7h', 'PPO8h', 'PPO9h', 'PPO10', 'PPO11', 'POP1h', 'POP2h', 'POP3h',
                         'POP4h', 'POP5h', 'POP6h', 'POP7h', 'POP8h', 'POP9h', 'POP10', 'POP11', 'POO1h', 'POO2h',
                         'POO3h', 'POO4h', 'POO5h', 'POO6h', 'POO7h', 'POO8h', 'POO9h', 'POO10', 'POO11', 'OPO1h',
                         'OPO2h', 'OPO3h', 'OPO4h', 'OPO5h', 'OPO6h', 'OPO7h', 'OPO8h', 'OPO9h', 'OPO10', 'OPO11',
                         'AFC1h', 'AFC2h', 'AFC3h', 'AFC4h', 'AFC5h', 'AFC6h', 'AFC7h', 'AFC8h', 'AFC9h', 'AFC10',
                         'AFC11', 'CAF1h', 'CAF2h', 'CAF3h', 'CAF4h', 'CAF5h', 'CAF6h', 'CAF7h', 'CAF8h', 'CAF9h',
                         'CAF10', 'CAF11', 'FFT1h', 'FFT2h', 'FFT3h', 'FFT4h', 'FFT5h', 'FFT6h', 'FFT7h', 'FFT8h',
                         'FFT9h', 'FFT10', 'FFT11', 'FTF1h', 'FTF2h', 'FTF3h', 'FTF4h', 'FTF5h', 'FTF6h', 'FTF7h',
                         'FTF8h', 'FTF9h', 'FTF10', 'FTF11', 'FTT1h', 'FTT2h', 'FTT3h', 'FTT4h', 'FTT5h', 'FTT6h',
                         'FTT7h', 'FTT8h', 'FTT9h', 'FTT10', 'FTT11', 'TFT1h', 'TFT2h', 'TFT3h', 'TFT4h', 'TFT5h',
                         'TFT6h', 'TFT7h', 'TFT8h', 'TFT9h', 'TFT10', 'TFT11', 'TTP1h', 'TTP2h', 'TTP3h', 'TTP4h',
                         'TTP5h', 'TTP6h', 'TTP7h', 'TTP8h', 'TTP9h', 'TTP10', 'TTP11', 'TPP1h', 'TPP2h', 'TPP3h',
                         'TPP4h', 'TPP5h', 'TPP6h', 'TPP7h', 'TPP8h', 'TPP9h', 'TPP10', 'TPP11', 'OI10h', 'OI11h',
                         'Fp1h', 'Fp2h', 'Fp3h', 'Fp4h', 'Fp5h', 'Fp6h', 'Fp7h', 'Fp8h', 'Fp9h', 'Fp10', 'Fp11', 'F10h',
                         'F11h', 'C10h', 'C11h', 'P10h', 'P11h', 'O10h', 'O11h', 'T10h', 'T11h', 'I10h', 'I11h', 'AFpz',
                         'AFp1', 'AFp2', 'AFp3', 'AFp4', 'AFp5', 'AFp6', 'AFp7', 'AFp8', 'AFp9', 'AF1h', 'AF2h', 'AF3h',
                         'AF4h', 'AF5h', 'AF6h', 'AF7h', 'AF8h', 'AF9h', 'AF10', 'AF11', 'FC1h', 'FC2h', 'FC3h', 'FC4h',
                         'FC5h', 'FC6h', 'FC7h', 'FC8h', 'FC9h', 'FC10', 'FC11', 'CP1h', 'CP2h', 'CP3h', 'CP4h', 'CP5h',
                         'CP6h', 'CP7h', 'CP8h', 'CP9h', 'CP10', 'CP11', 'PO1h', 'PO2h', 'PO3h', 'PO4h', 'PO5h', 'PO6h',
                         'PO7h', 'PO8h', 'PO9h', 'PO10', 'PO11', 'FT1h', 'FT2h', 'FT3h', 'FT4h', 'FT5h', 'FT6h', 'FT7h',
                         'FT8h', 'FT9h', 'FT10', 'FT11', 'TP1h', 'TP2h', 'TP3h', 'TP4h', 'TP5h', 'TP6h', 'TP7h', 'TP8h',
                         'TP9h', 'TP10', 'TP11', 'AFFz', 'AFF1', 'AFF2', 'AFF3', 'AFF4', 'AFF5', 'AFF6', 'AFF7', 'AFF8',
                         'AFF9', 'FAFz', 'FAF1', 'FAF2', 'FAF3', 'FAF4', 'FAF5', 'FAF6', 'FAF7', 'FAF8', 'FAF9', 'FFCz',
                         'FFC1', 'FFC2', 'FFC3', 'FFC4', 'FFC5', 'FFC6', 'FFC7', 'FFC8', 'FFC9', 'FCFz', 'FCF1', 'FCF2',
                         'FCF3', 'FCF4', 'FCF5', 'FCF6', 'FCF7', 'FCF8', 'FCF9', 'FCCz', 'FCC1', 'FCC2', 'FCC3', 'FCC4',
                         'FCC5', 'FCC6', 'FCC7', 'FCC8', 'FCC9', 'CFCz', 'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6',
                         'CFC7', 'CFC8', 'CFC9', 'CCPz', 'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
                         'CCP9', 'CPCz', 'CPC1', 'CPC2', 'CPC3', 'CPC4', 'CPC5', 'CPC6', 'CPC7', 'CPC8', 'CPC9', 'CPPz',
                         'CPP1', 'CPP2', 'CPP3', 'CPP4', 'CPP5', 'CPP6', 'CPP7', 'CPP8', 'CPP9', 'PCPz', 'PCP1', 'PCP2',
                         'PCP3', 'PCP4', 'PCP5', 'PCP6', 'PCP7', 'PCP8', 'PCP9', 'PPOz', 'PPO1', 'PPO2', 'PPO3', 'PPO4',
                         'PPO5', 'PPO6', 'PPO7', 'PPO8', 'PPO9', 'POPz', 'POP1', 'POP2', 'POP3', 'POP4', 'POP5', 'POP6',
                         'POP7', 'POP8', 'POP9', 'POOz', 'POO1', 'POO2', 'POO3', 'POO4', 'POO5', 'POO6', 'POO7', 'POO8',
                         'POO9', 'OPOz', 'OPO1', 'OPO2', 'OPO3', 'OPO4', 'OPO5', 'OPO6', 'OPO7', 'OPO8', 'OPO9', 'AFCz',
                         'AFC1', 'AFC2', 'AFC3', 'AFC4', 'AFC5', 'AFC6', 'AFC7', 'AFC8', 'AFC9', 'CAFz', 'CAF1', 'CAF2',
                         'CAF3', 'CAF4', 'CAF5', 'CAF6', 'CAF7', 'CAF8', 'CAF9', 'FFTz', 'FFT1', 'FFT2', 'FFT3', 'FFT4',
                         'FFT5', 'FFT6', 'FFT7', 'FFT8', 'FFT9', 'FTFz', 'FTF1', 'FTF2', 'FTF3', 'FTF4', 'FTF5', 'FTF6',
                         'FTF7', 'FTF8', 'FTF9', 'FTTz', 'FTT1', 'FTT2', 'FTT3', 'FTT4', 'FTT5', 'FTT6', 'FTT7', 'FTT8',
                         'FTT9', 'TFTz', 'TFT1', 'TFT2', 'TFT3', 'TFT4', 'TFT5', 'TFT6', 'TFT7', 'TFT8', 'TFT9', 'TTPz',
                         'TTP1', 'TTP2', 'TTP3', 'TTP4', 'TTP5', 'TTP6', 'TTP7', 'TTP8', 'TTP9', 'TPPz', 'TPP1', 'TPP2',
                         'TPP3', 'TPP4', 'TPP5', 'TPP6', 'TPP7', 'TPP8', 'TPP9', 'OI1h', 'OI2h', 'OI3h', 'OI4h', 'OI5h',
                         'OI6h', 'OI7h', 'OI8h', 'OI9h', 'OI10', 'OI11', 'Fpz', 'Fp1', 'Fp2', 'Fp3', 'Fp4', 'Fp5',
                         'Fp6', 'Fp7', 'Fp8', 'Fp9', 'F1h', 'F2h', 'F3h', 'F4h', 'F5h', 'F6h', 'F7h', 'F8h', 'F9h',
                         'F10', 'F11', 'C1h', 'C2h', 'C3h', 'C4h', 'C5h', 'C6h', 'C7h', 'C8h', 'C9h', 'C10', 'C11',
                         'P1h', 'P2h', 'P3h', 'P4h', 'P5h', 'P6h', 'P7h', 'P8h', 'P9h', 'P10', 'P11', 'O1h', 'O2h',
                         'O3h', 'O4h', 'O5h', 'O6h', 'O7h', 'O8h', 'O9h', 'O10', 'O11', 'T1h', 'T2h', 'T3h', 'T4h',
                         'T5h', 'T6h', 'T7h', 'T8h', 'T9h', 'T10', 'T11', 'I1h', 'I2h', 'I3h', 'I4h', 'I5h', 'I6h',
                         'I7h', 'I8h', 'I9h', 'I10', 'I11', 'AFz', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7',
                         'AF8', 'AF9', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FC7', 'FC8', 'FC9', 'CPz',
                         'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CP7', 'CP8', 'CP9', 'POz', 'PO1', 'PO2', 'PO3',
                         'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'PO9', 'FTz', 'FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6',
                         'FT7', 'FT8', 'FT9', 'TPz', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'TP6', 'TP7', 'TP8', 'TP9',
                         'OIz', 'OI1', 'OI2', 'OI3', 'OI4', 'OI5', 'OI6', 'OI7', 'OI8', 'OI9', 'LO1', 'LO2', 'IO1',
                         'SO1', 'IO2', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'Cz', 'C1', 'C2',
                         'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                         'P9', 'Oz', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'Tz', 'T1', 'T2', 'T3', 'T4',
                         'T5', 'T6', 'T7', 'T8', 'T9', 'Iz', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'Nz'
                         ]

OTHER_EEG_SYSTEMS = {
        "R08", "R07", "R06", "R05", "R04", "R03", "R02", "R01", "L08", "L07", "L06", "L05", "L04b", "L04a",
        "L04", "L03", "L02", "L01", 'M1', 'M2'}


EEG_PREFIXES_SUFFIXES = {"EEG", "FP", "REF", "LE", "RE"}
