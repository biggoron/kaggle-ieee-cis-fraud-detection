# TODO: compute
FRAUD_MEAN = 6.30051906080931
FRAUD_FREQ = 6.951546426196875e-05
FRAUD_AMP = 5.400106899195011

FIRST_NAN_GROUP = [
    'V316', 'V294', 'V295', 'V297', 'V298', 'V299', 'V302',
    'V303', 'V304', 'V317', 'V306', 'V307', 'V308', 'V309',
    'V310', 'V311', 'V312', 'V321', 'V320', 'V319', 'V293',
    'V292', 'V305', 'V318', 'V290', 'V291', 'V287', 'V286',
    'V285', 'V284', 'V279', 'V280']

SECOND_NAN_GROUP = [
    'V119', 'V123', 'V122', 'V121', 'V120', 'V118', 'V117',
    'V116', 'V115', 'V114', 'V113', 'V112', 'V111', 'V124',
    'V110', 'V108', 'V107', 'V106', 'V105', 'V104', 'V103',
    'V102', 'V101', 'V100', 'V99', 'V98', 'V97', 'V109', 'V96',
    'V125', 'V127', 'V126', 'V137', 'V136', 'V135', 'V134',
    'V95', 'V132', 'V131', 'V130', 'V129', 'V128', 'V133']

THIRD_NAN_GROUP = [
    'V283', 'V313', 'V301', 'V300', 'V296', 'V289', 'D1',
    'V288', 'V314', 'V315', 'V281', 'V282']

FOURTH_NAN_GROUP = [
    'D10', 'V29', 'V32', 'V31', 'V30', 'V28', 'V27', 'V26',
    'V25', 'V24', 'V23', 'V22', 'V21', 'V20', 'V19', 'V12',
    'V13', 'V14', 'V15', 'V16', 'V33', 'V34', 'V18', 'V17']

FIFTH_NAN_GROUP = [
    'V54', 'V74', 'V73', 'V72', 'V71', 'V70', 'V69', 'V68',
    'V67', 'V66', 'V65', 'V63', 'V62', 'V60', 'V59', 'V58',
    'V57', 'V56', 'V55', 'V64', 'V53', 'V61']

SIXTH_NAN_GROUP = [
    'D15', 'V84', 'V82', 'V81', 'V80', 'V79', 'V78', 'V77', 'V76',
    'V75', 'V83', 'V86', 'V87', 'V85', 'V93', 'V92', 'V88', 'V94',
    'V91', 'V90', 'V89']

SEVENTH_NAN_GROUP = [
    'D4', 'V39', 'V41', 'V38', 'V46', 'V36', 'V35', 'V44', 'V45',
    'V40', 'V37', 'V50', 'V48', 'V49', 'V42', 'V51', 'V52', 'V47',
    'V43']

EIGTH_NAN_GROUP = [
    'D11', 'V10', 'V11', 'V2', 'V3', 'V4', 'V1', 'V6', 'V8', 'V9',
    'V5', 'V7']

NINTH_NAN_GROUP = [
    'V270', 'V271', 'V250', 'V220', 'V222', 'V255', 'V227', 'V251',
    'V259', 'V256', 'V272', 'V221', 'V239', 'V234', 'V245', 'V238']

ELEVENTH_NAN_GROUP = [
    'V184', 'V201', 'V171', 'V208', 'V209', 'V210', 'V174', 'V175',
    'V180', 'V170', 'V185', 'V188', 'V169', 'V200', 'V198', 'V197',
    'V195', 'V194', 'V189', 'V190', 'V199', 'V172', 'V196', 'V173',
    'V176', 'V177', 'V179', 'V193', 'V203', 'V182', 'V183', 'V192',
    'V191', 'V186', 'V187', 'V178', 'V181', 'V205', 'V167', 'V168',
    'V216', 'V215', 'V214', 'V213', 'V202', 'V211', 'V207', 'V206',
    'V204', 'V212']

TWELVTH_NAN_GROUP = [
    'V218', 'V217', 'V240', 'V241', 'V253', 'V243', 'V219', 'V244',
    'V246', 'V242', 'V223', 'V226', 'V225', 'V247', 'V228', 'V229',
    'V230', 'V231', 'V232', 'V233', 'V235', 'V224', 'V248', 'V257',
    'V252', 'V278', 'V277', 'V276', 'V275', 'V274', 'V273', 'V269',
    'V268', 'V249', 'V267', 'V265', 'V264', 'V263', 'V262', 'V261',
    'V260', 'V258', 'V236', 'V254', 'V266', 'V237']

THIRTEENTH_NAN_GROUP = [
    'V332', 'V333', 'V322', 'V323', 'V324', 'V325', 'V326', 'V328',
    'V327', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'V330',
    'V329', 'V331']

FOURTEENTH_NAN_GROUP = [
    'V159', 'V165', 'V164', 'V160', 'V166', 'V151', 'V150', 'V143',
    'V144', 'V145', 'V152'] 

FIFTEENTH_NAN_GROUP = [
    'V138', 'V139', 'V140', 'V141', 'V142', 'V146', 'V147', 'V163',
    'V162', 'V161', 'V149', 'V158', 'V157', 'V156', 'V155', 'V154',
    'V148', 'V153']

EMAILS = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
          'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',
          'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other',
          'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',
          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
          'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',
          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
US_EMAILS = ['gmail', 'net', 'edu']

