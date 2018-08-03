import smtplib
from email.mime.text import MIMEText
from email.header import Header
import pandas as pd
import numpy as np

SERVERSPOT = 'MAC' # 'AMS', 'MAC'
ROOTPATH = '/home/ubuntu/notebooks/Stock-Complex-Networks'
if SERVERSPOT == 'MAC': ROOTPATH = '/Users/zhanglingjie/Documents/GitHub/Stock-Complex-Networks'

DROPLIST = ['ASC', 'BLX', 'BP', 'CHKP', 'CPA', 'DHT', 'DLPH', 'DOX', 'EROS', 'FRO', 'GPT', 'I', 'LSI', 'LXFT', 'MIND', 'NM', 'NNA', 'PERI', 'QGEN', 'SALT', 'SB', 'SBNY', 'SDRL', 'SFL', 'SPNS', 'SSYS', 'STNG', 'TGH', 'TK', 'TRI', 'TSRA', 'VOD', 'ZBH', 'ACAD', 'ACRX', 'ACTG', 'ADMS', 'ADSK', 'AERI', 'AGEN', 'AGIO', 'AGYS', 'AHC', 'AKAO', 'AKBA', 'AKS', 'ALDR', 'ALIM', 'ALKS', 'ALNY', 'AMBC', 'AMBR', 'AMCC', 'AMD', 'AMPE', 'AMRI', 'AMSC', 'ANGO', 'APA', 'APC', 'ARAY', 'AREX', 'ARIA', 'ARNA', 'ARQL', 'ARR', 'ARWR', 'ATEC', 'ATI', 'ATRC', 'ATRS', 'AVEO', 'AVNW', 'AVP', 'AXAS', 'AXDX', 'BAS', 'BBG', 'BCOV', 'BCRX', 'BDSI', 'BGC', 'BIOL', 'BIOS', 'BLUE', 'BMRN', 'BNFT', 'BPTH', 'BRS', 'BV', 'CALD', 'CALX', 'CARA', 'CARB', 'CAVM', 'CCXI', 'CDI', 'CENX', 'CERS', 'CETV', 'CHK', 'CIDM', 'CKH', 'CLDX', 'CLNE', 'CLR', 'CLVS', 'CMA', 'CMRX', 'CNX', 'COG', 'COP', 'COVS', 'CPE', 'CRIS', 'CRK', 'CROX', 'CRZO', 'CSC', 'CSII', 'CSLT', 'CSOD', 'CTIC', 'CTT', 'CUI', 'CUR', 'CWEI', 'CWST', 'CY', 'CYTR', 'CYTX', 'DATA', 'DAVE', 'DDD', 'DEPO', 'DEST', 'DHX', 'DMRC', 'DNR', 'DRNA', 'DSCI', 'DVAX', 'DVN', 'DWSN', 'DXCM', 'DXLG', 'DXYN', 'ECOM', 'ECYT', 'EGAN', 'EGHT', 'EGL', 'EGLT', 'EGN', 'EGY', 'ELGX', 'ENDP', 'ENPH', 'ENT', 'EOG', 'EPAY', 'EPZM', 'ESPR', 'EXAS', 'EXEL', 'FANG', 'FBRC', 'FCH', 'FCSC', 'FET', 'FIVN', 'FLDM', 'FLXN', 'FMI', 'FOLD', 'FPO', 'FSTR', 'FTD', 'FTR', 'FVE', 'GALT', 'GEOS', 'GFN', 'GHDX', 'GLF', 'GLUU', 'GNCA', 'GNMK', 'GNW', 'GOGO', 'GOOD', 'GPOR', 'GST', 'GTXI', 'GUID', 'HAL', 'HALO', 'HBIO', 'HEAR', 'HES', 'HHS', 'HIVE', 'HK', 'HLIT', 'HLX', 'HMHC', 'HRTX', 'ICPT', 'IDRA', 'IL', 'IMGN', 'IMI', 'IMMU', 'IMPV', 'INO', 'INSM', 'INTX', 'IO', 'IRWD', 'ITCI', 'IVAC', 'IVC', 'JIVE', 'JOY', 'KEG', 'KERX', 'KEYW', 'KIN', 'KMT', 'KOPN', 'KOS', 'KPTI', 'KTWO', 'LIFE', 'LL', 'LLNW', 'LMIA', 'LNG', 'LPI', 'LPSN', 'LQDT', 'LSCC', 'LTS', 'LUB', 'LXRX', 'MACK', 'MCF', 'MDCA', 'MDCO', 'MEG', 'MEIP', 'MGNX', 'MNI', 'MNTA', 'MNTX', 'MOSY', 'MRC', 'MRIN', 'MRO', 'MRTX', 'MTDR', 'MUR', 'MXL', 'NAV', 'NAVB', 'NBIX', 'NBL', 'NBR', 'NDLS', 'NEO', 'NEON', 'NETE', 'NFG', 'NFX', 'NKTR', 'NLNK', 'NMBL', 'NMRX', 'NNVC', 'NOV', 'NOW', 'NR', 'NRG', 'NSTG', 'NUAN', 'NVAX', 'NWPX', 'NWY', 'NXTM', 'NYRT', 'OAS', 'OCN', 'OHRP', 'OMED', 'OMER', 'OPHT', 'ORBC', 'OREX', 'OSIR', 'OVAS', 'OXFD', 'OXY', 'OZRK', 'P', 'PACB', 'PANW', 'PBYI', 'PCTI', 'PCTY', 'PCYO', 'PDCE', 'PE', 'PEGI', 'PEI', 'PES', 'PETX', 'PFIE', 'PFPT', 'PHH', 'PHX', 'PKD', 'PODD', 'PQ', 'PRKR', 'PRO', 'PRTA', 'PTCT', 'PTEN', 'PTIE', 'PTLA', 'PTX', 'PXD', 'QDEL', 'QEP', 'QNST', 'QRHC', 'QTM', 'QTWO', 'QUIK', 'RARE', 'RATE', 'RBCN', 'RCII', 'REI', 'RELL', 'REN', 'RES', 'REXX', 'RFP', 'RGLS', 'RIGL', 'RMTI', 'RNET', 'RNG', 'RNWK', 'RPRX', 'RRC', 'RSPP', 'RST', 'RSYS', 'RT', 'RVNC', 'S', 'SBY', 'SCOR', 'SEAC', 'SFE', 'SGEN', 'SGM', 'SGMO', 'SGMS', 'SGYP', 'SIF', 'SM', 'SNMX', 'SNSS', 'SPLK', 'SPN', 'SPNC', 'SPPI', 'SPRT', 'SREV', 'SRPT', 'STAA', 'STML', 'SWN', 'SYX', 'TAT', 'TBPH', 'TCBI', 'TEAR', 'TESO', 'TGTX', 'THC', 'TIME', 'TNAV', 'TNDM', 'TNGO', 'TPLM', 'TRXC', 'TSLA', 'TSRO', 'TTI', 'TTPH', 'TWI', 'TWOU', 'TXMD', 'UIS', 'UMH', 'UNIS', 'UNT', 'UNXL', 'USAP', 'VC', 'VCRA', 'VCYT', 'VECO', 'VHC', 'VICL', 'VNDA', 'VRNS', 'VRTX', 'VRX', 'VSAR', 'VSTM', 'VTL', 'VTNR', 'WDAY', 'WIFI', 'WLB', 'WLL', 'WMB', 'WMC', 'WMGI', 'WPX', 'WSTL', 'WTI', 'X', 'XEC', 'XLRN', 'XON', 'XONE', 'XRM', 'YUME', 'Z', 'ZEN', 'ZEUS', 'ZIOP', 'ZNGA', 'AIV', 'AON', 'CACQ', 'EIX', 'HNR', 'LGIH', 'LORL', 'LVLT', 'MN', 'SGY', 'AIQ', 'AVID', 'AZPN', 'BXC', 'CBB', 'CCOI', 'CHH', 'CL', 'CLUB', 'CUDA', 'DENN', 'EAT', 'FFNW', 'FRP', 'HCA', 'HOV', 'LB', 'LEE', 'MAA', 'MGI', 'MNKD', 'MSI', 'MTOR', 'NATH', 'PENN', 'PM', 'PROV', 'QTS', 'REG', 'REV', 'SBAC', 'SIRI', 'SN', 'SONC', 'SVU', 'TDG', 'VGR', 'VRSN', 'WSTC', 'XOMA', 'YRCW', 'ACGL', 'AF', 'AGII', 'AGN', 'AGO', 'AHL', 'AIRM', 'ALJ', 'ALLE', 'ALR', 'ATW', 'AXS', 'BCR', 'BEAV', 'BOBE', 'BWLD', 'CACB', 'CALL', 'CB', 'CBF', 'CCE', 'CFNL', 'CPN', 'CSBK', 'CUNB', 'CWCO', 'ESGR', 'ESNT', 'ESV', 'FN', 'FTI', 'G', 'GBLI', 'GLRE', 'GRMN', 'HEOP', 'HZNP', 'IR', 'IXYS', 'LAZ', 'LBTYA', 'LDR', 'LIOX', 'MBRG', 'MBVT', 'MDT', 'MENT', 'MHLD', 'MNK', 'MRVL', 'MSFG', 'NE', 'NLSN', 'NRCIA', 'OKSB', 'PCBK', 'PNRA', 'PRGO', 'PVTB', 'RE', 'RIG', 'RNR', 'SCLN', 'SCMP', 'SHOR', 'SIG', 'SNBC', 'SNI', 'STE', 'STX', 'TEL', 'TPRE', 'VASC', 'VR', 'WBC', 'WBMD', 'WCN', 'WTM', 'XL', 'ACN']

###
def Polygonal(n, sides):
    return (n**2*(sides-2) - n*(sides-4)) // 2

def Triangular(n):
    return Polygonal(n, 3)
###

def LocateIdx(x, y, num_of_obj):
    if x >= y:
        return -1
    return int((2*num_of_obj-x-1)*x/2 + y - x)

###
def PartialTriangular(n, num_of_obj):
    return int((2*num_of_obj-n-1)*n/2)

def DcpsIdx(idx, num_of_obj):
    for i in range(LENTCKR):
        if idx <= PartialTriangular(i, num_of_obj):
            return int(i-1), int(idx - PartialTriangular(i-1, num_of_obj) + i - 1)
    return -1
###

def SendEmail(Msg=None):
    gmail_user = 'zljlwty@qq.com'
    gmail_password = 'stlsfgfqytumbcgg'
    sent_from = gmail_user
    to = ['zspling@gmail.com']
    subject = 'Program Progress Inform'
    body = '\r\n'.join([
      'From: ' + sent_from,
      'To: ' + to[0],
      'Subject: ' + subject,
      '',
      'Done!\n' + str(Msg)
      ])
    try:
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, body)
        server.close()
        print('Email sent!')
    except:
        print(sys.exc_info()[0])

#FILE_TICKER_FDMTL = ROOTPATH + r'/Source/DF_FDMTL_0612.csv'
#df_fdmtl = pd.read_csv(FILE_TICKER_FDMTL).set_index('ticker')

#FILE_DF_SOURCE = ROOTPATH + r'/Source/DF_SOURCE_0613.csv'
#df_source = pd.read_csv(FILE_DF_SOURCE).set_index('ticker')

FILE_SP500 = ROOTPATH + r'/Source/^GSPC.csv'
df_SP500 = pd.read_csv(FILE_SP500).set_index('Date')
sri_SP500_close = df_SP500['Close']
sri_SP500_log_return = np.log(sri_SP500_close / sri_SP500_close.shift())
sri_SP500_log_return = sri_SP500_log_return.drop(sri_SP500_log_return.index[0])
