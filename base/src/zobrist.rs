use crate::{Color, Piece, Square};

#[inline(always)]
/// Get the Zobrist key for a given key, type, and square.
pub fn square_key(sq: Square, pt: Option<Piece>, color: Color) -> u64 {
    match pt {
        None => 0,
        // Because sq, p, and color are all enums with fixed ranges, we can
        // perform an unchecekd get on these indices.
        Some(p) => unsafe {
            *SQUARE_KEYS
                .get_unchecked(sq as usize)
                .get_unchecked(p as usize)
                .get_unchecked(color as usize)
        },
    }
}

#[inline(always)]
/// Get the Zobrist key for a castling right. 0 is for white king castle, 1 is
/// for white queen castle, 2 is for black king castle, and 3 is for black
/// queen castle.
pub const fn get_castle_key(right: u8) -> u64 {
    CASTLE_KEYS[right as usize]
}

#[inline(always)]
/// Get the Zobrist key of an en passant square.
pub fn ep_key(ep_square: Option<Square>) -> u64 {
    match ep_square {
        None => 0,
        // Since the square is in the square enum, we can safely get this
        // without checking.
        Some(sq) => unsafe { *EP_KEYS.get_unchecked(sq.file() as usize) },
    }
}

#[inline(always)]
/// Get the Zobrist key for the player to move
pub const fn player_to_move_key(player_to_move: Color) -> u64 {
    match player_to_move {
        Color::White => 0,
        Color::Black => BLACK_TO_MOVE_KEY,
    }
}

/// A set of random keys which will reliably be the same.
/// If one wanted, they could generate a new set of keys on every program run,
/// but that seems inefficient.
const SQUARE_KEYS: [[[u64; 2]; Piece::NUM_TYPES]; 64] = [
    [
        [13515657874892102023, 15129553140981592645],
        [14809938836708178893, 15774518201988393282],
        [17188393767951940199, 4985137385606203988],
        [8046993897209325893, 6881813287187799947],
        [16125670290659210324, 4816260449376201035],
        [11570069662165393444, 11285152104292927369],
    ],
    [
        [3446018582270436793, 13616437186283622689],
        [13519059396826741711, 9796009710311115986],
        [13565560346403939986, 10792308714086901791],
        [3072727624150280346, 3794756855380371525],
        [1435931999638272781, 15441221205979428565],
        [5940567713233535239, 8464077112431967362],
    ],
    [
        [495664619263271681, 9420932444907695650],
        [12741368234109995270, 7664225158426101066],
        [123100759219922640, 12246793704608856168],
        [3021706240337810846, 10701126688782509631],
        [5286375280603737045, 12037244071987171008],
        [1953382953708866778, 12569849586622039179],
    ],
    [
        [3350919833121026612, 6917572908190574316],
        [17712016599341141622, 10994228475909012897],
        [16844352835873823081, 6085447665261760849],
        [742500085444621992, 2209962125935807334],
        [9209073793829003437, 13598795564570686766],
        [13946642687760876726, 12353966073431331312],
    ],
    [
        [4754343899128033937, 7999896076212440554],
        [16897746622936208918, 2395661151836497898],
        [18073709211994443880, 5558496187847604090],
        [15948151388970225971, 172597821324353835],
        [17801296389112086916, 6451330536465442251],
        [6308175451005450063, 7120683906288837975],
    ],
    [
        [236895846088702896, 1123221273390886733],
        [8210899693915372461, 7159189460286746686],
        [14740380003831013958, 3310910749645895425],
        [215996133496893554, 13140254967128155258],
        [3972520063736808413, 9454387687692677134],
        [2241369823883942596, 11471090748910479112],
    ],
    [
        [254892419357612817, 9071938695347067170],
        [14240662866761158599, 17820701860784163862],
        [17019042975767554549, 10142200304454143677],
        [6719910664025650302, 12227359393470203212],
        [2202202444884407082, 2760730179108792751],
        [9129165348354779568, 16463810222779680359],
    ],
    [
        [6189910006475770528, 3749341880136721304],
        [3002308468479489223, 2769677883324350151],
        [1671196606386453292, 12833774114208833848],
        [2703214826562935424, 12132134214268140228],
        [16342691665418270362, 15407886221016473523],
        [17225736924774608060, 15691010154271609526],
    ],
    [
        [4875081210458220180, 13835488679462871749],
        [1219633687171738399, 13861914381804840313],
        [8501463582677749277, 3356156007447137140],
        [16951352228835675164, 3400123950416765309],
        [6879041868768447104, 3245900172204009703],
        [3280796154694990799, 10544962268059163078],
    ],
    [
        [15190462450199705153, 16479028770895632643],
        [17712200011694641618, 9466971872010357883],
        [7702114492101903268, 2216293578142963653],
        [11874895798415470047, 13716350736066119448],
        [11587208491360567030, 10537423829264651920],
        [13819369558524233243, 9794525146463145015],
    ],
    [
        [4302523723566195431, 10475242094662107668],
        [2312500128173556785, 9413901950967830200],
        [14229509250589227890, 16444977514914674576],
        [1092094222684883897, 14336988344293731611],
        [812394947067494019, 14743184481622720604],
        [9885644567209155075, 1984249643104776717],
    ],
    [
        [13997893337761777888, 2880890189287616219],
        [16239442326123452825, 11710128666576187848],
        [5071563865067304343, 8191986895770187453],
        [6801043646299192381, 5667842744804691012],
        [11259406877500171085, 5752592898619856201],
        [9085165825001424155, 929057713734539710],
    ],
    [
        [8219098178975139502, 4051554117729101870],
        [2510049438397274779, 37539989512659499],
        [17885833627067471158, 14841920734294932415],
        [1533724647794856334, 2048711013915015826],
        [16088201148346570085, 15918573569284344452],
        [9048644028699911106, 5029010618217972638],
    ],
    [
        [3847131426912768470, 5536765039475735656],
        [5816357415300583143, 7162469989669573156],
        [11378905987195004573, 6795882747517997663],
        [15956024853442510940, 7733943280753902464],
        [2961185793115268907, 2635017936014971198],
        [5654153221234187655, 6139674184629124032],
    ],
    [
        [3706048374401557226, 2765206443233383179],
        [17555890547596017841, 581555062465446482],
        [18394352343749710840, 7233658005278787601],
        [15983800366294684311, 4452266131179350573],
        [8939982845187747816, 10669419923270603356],
        [4780555328709021645, 9820253591571635384],
    ],
    [
        [2921705737197482145, 13673504011398697627],
        [701214022870724154, 16075435385646555774],
        [1733730555209352268, 3824522835797285545],
        [9300857857734498685, 17733362906453806440],
        [18091059300905431170, 280646307971082086],
        [9592133770406705170, 10854810339779764535],
    ],
    [
        [6001800049176622608, 14065975969455046841],
        [16602061026503773396, 8438916352800520713],
        [8109449197853496498, 6857602186584140873],
        [504565317601412469, 7165342442415664776],
        [12435925833963587604, 9978008497326312433],
        [10962506466513424050, 5888892666789012334],
    ],
    [
        [8073093768810702277, 5884743457921404278],
        [7184599454185378354, 5505776892952096457],
        [7004556387130681357, 11483647489890586274],
        [10519422850103058427, 8968319072355583688],
        [2189576784016640528, 11340773966255892325],
        [613541993151380931, 944746977838023604],
    ],
    [
        [15403046205723302446, 9685912476233493576],
        [8493731688670646629, 8653313068594333475],
        [7809539545670843967, 16657864039519841112],
        [6078611127004872991, 14472837809762854105],
        [7068805004932505562, 10021674466772267409],
        [18433059359754779580, 12655079871591682036],
    ],
    [
        [8698923853861779317, 3328150744091176177],
        [3845570127062622223, 10595356001425089451],
        [10167046230606908870, 3945370502885661952],
        [12410271439419987436, 8401590340661490346],
        [14001652878199674597, 12001600185277028406],
        [1350534806158850049, 16596650780766233983],
    ],
    [
        [1286641337526839560, 14902014645110978261],
        [1742545447434564826, 10456504947366666880],
        [3940049762519602869, 5121248166771395387],
        [9180397479822025170, 13543430662236689210],
        [8240775069717263661, 7444115909587059783],
        [4595697880528574621, 8312132785784667826],
    ],
    [
        [15441654137287150866, 9305172862618369250],
        [9557846983478291839, 9460913230583332088],
        [9964955357692366853, 12560140615587145336],
        [4405694664027662415, 9098226866343546155],
        [1520347936722403613, 8588021447873005033],
        [6046045842675880505, 8194311226856896478],
    ],
    [
        [7833101518264898751, 2298587622176603340],
        [4020732659928169577, 11379043003751970156],
        [16983251254711162722, 4837154683380640805],
        [16248092391112713753, 14339567456591350213],
        [3246696302234862481, 103286982553998369],
        [13506180587600915473, 5096508690172926085],
    ],
    [
        [14874368531026464522, 8841811441281811968],
        [6129842650868402849, 11643452713741453620],
        [14579780478759161546, 5574861350962174201],
        [13413056971447411005, 6770407480130249139],
        [17269501669174047737, 18188846021733073843],
        [4250985302956179630, 16971576073837852559],
    ],
    [
        [3763658382942842373, 16939301748933350815],
        [107905296164793381, 2791638973359621664],
        [10736653257205027830, 17132756303845540535],
        [2459090258910070954, 3932657329632269004],
        [15451026766375622090, 11492066180030965451],
        [9411561845631174232, 534866473465365570],
    ],
    [
        [7908754111196696099, 16999508699883027622],
        [10718385396005537231, 14590866674972069064],
        [11813628010819326433, 12184185840868700487],
        [8119493788028810486, 10375735932604079359],
        [5164723878777677434, 5210646178110629002],
        [1349105120390208534, 8804530187640763946],
    ],
    [
        [15673105961070325757, 8901426037300317133],
        [11970601808154697059, 16589808676133281381],
        [876317825781207391, 1813831521813494383],
        [16601368196528947961, 2199485887224225382],
        [12239081181696491555, 16379232572083880035],
        [13502142480378247063, 8795531688847485970],
    ],
    [
        [18102865503213786336, 12698472168401885601],
        [2235216999254036599, 7948703970817338580],
        [18267287593607045447, 134573010360514964],
        [8464256675254656746, 11565019891275682502],
        [6566172802782242580, 15524123158519870048],
        [4978072196553269461, 2664517733480094252],
    ],
    [
        [2958856770638028723, 5949157256856541194],
        [11083321508415952377, 7930386204064851961],
        [6533480499674460970, 12741221540646545547],
        [15919246764230384755, 493237589586897642],
        [12383349438457732815, 9183159247589135628],
        [11198100388422760695, 14113298014381838647],
    ],
    [
        [15934051970793380050, 15386918489199617820],
        [15239102331574997518, 5509566918550206628],
        [10099183947161525243, 9203019853854518954],
        [12763790276117431443, 8601208915559612456],
        [6780983172911701659, 7553101063643132431],
        [8703117704682386216, 11376243546202152900],
    ],
    [
        [6272939414622188789, 5450689546596956289],
        [17897802172562405692, 4216840327318547606],
        [7663291433122506876, 16869665266908524255],
        [2765922899972110667, 9847447312795365027],
        [14031058935278551791, 18175924862358787498],
        [17453796359929960392, 2267931409147592630],
    ],
    [
        [7138666146762821487, 6834682368918972135],
        [3696875117279462373, 3967384405088497079],
        [1764378947523126168, 4811536208277120330],
        [4593264373642890577, 9929253848944581970],
        [5107714770498734607, 11316391444875009165],
        [10570943142834525082, 6395675079283270295],
    ],
    [
        [7225631690543289168, 10717825642964716181],
        [10859736714381746091, 8121284487145075065],
        [5820422365263660124, 4706022536554844020],
        [3158223615069289642, 5348529936361093103],
        [1122164482183171018, 2373596592903631057],
        [9736635378249847453, 12934992593475388290],
    ],
    [
        [3892097595260155385, 4707783840868912275],
        [6612226242069934727, 1468416874522005730],
        [12936264186145505797, 2607373434122262912],
        [13671484624356327130, 13128708559417648125],
        [823303345980353626, 15195905110619493303],
        [362830728866416504, 2655464836299302939],
    ],
    [
        [7828325570785338895, 7763241947939853480],
        [3194615233424363806, 2848370010695535135],
        [9408348941084041014, 2661626991153069221],
        [15124769082058958687, 6575168277164643048],
        [1381103538182975800, 2876872495088342636],
        [1325553428788532522, 15692295582185302837],
    ],
    [
        [2437627519503253004, 6656559208061837159],
        [17708913082212924222, 5433266215990364495],
        [9989030351405675046, 7485098168273597441],
        [6116212826296789066, 4635535061986947838],
        [17170737673241303322, 3128235923424980692],
        [13201735155304783172, 7095541244940188401],
    ],
    [
        [8196661389046893469, 16306201638950903996],
        [4725629266975527894, 14210250700740840048],
        [2849863034653146257, 13547085096086215747],
        [4664417134088968947, 12757628202723634230],
        [5346100451906631490, 42140834851339845],
        [10415380456202401148, 13124104947762760036],
    ],
    [
        [10202686012982704142, 6949208221954208997],
        [16605312558635699281, 2010439375914914245],
        [9947488007993687944, 8557877704156465544],
        [9256794552383623859, 1383642063300078741],
        [15416996531601540442, 2011929294337368934],
        [1988505359651277223, 2805404791919444030],
    ],
    [
        [13988429249875837714, 838224349064731661],
        [11740331219085023026, 7455685084570341409],
        [5268546906058063390, 9665883991103870303],
        [4153572744311269266, 16974011533812856963],
        [3252524679990046982, 1805994151105746382],
        [2717050386921169670, 4499577254254833410],
    ],
    [
        [2208648224880395821, 5792766379481664243],
        [7306883264871008204, 17728685252851219143],
        [13651250475069619964, 14498976773896711326],
        [3938321029711659861, 7203614769312454425],
        [96651561652652026, 17912310574082615195],
        [16457450998567200737, 13087226315628128030],
    ],
    [
        [7992581223028794087, 7719351088178957724],
        [18278346302268003320, 8751980422879675316],
        [4060605037368072287, 849231576586466956],
        [3221549073520347138, 3498768154926645165],
        [4036145003302584637, 13853230121667379468],
        [16662386502029266921, 18195968015465352452],
    ],
    [
        [10750558347962030091, 15907474856781998219],
        [10231551261551526604, 3273052469594833618],
        [3234884735583648689, 85937689850116791],
        [16242826259802871812, 2379457834277099277],
        [16813922645999255398, 13968013032604645270],
        [1082767903544515721, 2251081503002701850],
    ],
    [
        [9834287269475554643, 17650138086667977361],
        [15024395448744795088, 8882712123098147158],
        [10784898861257160825, 6243421988669985393],
        [1718143270825484936, 240743903574793443],
        [7598999826146039713, 5702442098101045836],
        [17996297225689436850, 1355161795365024067],
    ],
    [
        [17560458242179694719, 8322888833468142548],
        [17978220841715233862, 13646734040700887274],
        [2651461487074980060, 15302955340064392876],
        [6922728190880682523, 4584449031715800647],
        [15779558247807806086, 7979513383582368113],
        [15446839214308498228, 12980862144805318793],
    ],
    [
        [4154938442608612342, 17485626756426853574],
        [8517161989045981631, 2185904588660859395],
        [12877471233627355215, 16708705144763405688],
        [16518311926072498880, 2209783507418509436],
        [5041330832522948677, 9703323053277016405],
        [485461215193441123, 18235236799052115848],
    ],
    [
        [12095840423777072478, 18184851099831008588],
        [9121220867794951136, 3362248466797802259],
        [13830464331091546055, 7887328503881137541],
        [2717134609502116401, 3151262443840568004],
        [4307962245178361158, 2009875218980675013],
        [13418613826592614668, 12122022570206019005],
    ],
    [
        [3466480152039938200, 8242856444798062939],
        [9267801001438090435, 10395189655229790924],
        [7750966685779908642, 16965329916559200564],
        [994725560061531054, 12102582056912771849],
        [13215949942942206503, 17972379654432095712],
        [1789982290846440510, 8172750366980308379],
    ],
    [
        [5673254217238316540, 900992740804221313],
        [12209882543157665816, 15098865361431627652],
        [1917476431150067140, 14039115240650919338],
        [8245453499548560634, 14871774617356619943],
        [16630614936957425792, 1577206248623259209],
        [14968609283864283782, 8094127085592259981],
    ],
    [
        [17518038171145229764, 8086314533261227454],
        [18104481720601799881, 15829505680520714769],
        [6027228290869608748, 6336324607802349051],
        [10356429860658504735, 9012194578932714498],
        [11715026065550788032, 7821120926119460157],
        [3310752755662606342, 4585983460022049941],
    ],
    [
        [8890187558670924501, 14066173152019126861],
        [9882696763888466881, 6243843032320860218],
        [10031936855963008478, 14856031637580709544],
        [8902691283474891150, 18182545333852257005],
        [10169130617039757742, 5715725163630400628],
        [15788746051639665462, 6513264013417541906],
    ],
    [
        [3357401854089605336, 1376518414024477988],
        [8451313861535704190, 7999544650261526379],
        [14570752385596303000, 8317742259207405774],
        [13134168096352956684, 3763442495469020665],
        [4118160248925381498, 14111728018802483866],
        [15082780492858946742, 8473105375622261041],
    ],
    [
        [8037990081330665601, 7505317489120458065],
        [9688203659751990476, 14519939734062546734],
        [13115210026065458639, 13470918205317450871],
        [16897046651347172379, 15630042108408122864],
        [414826391596615030, 14779176993058006840],
        [4478932482000006396, 695455345245712315],
    ],
    [
        [6884952710633107382, 15721221223897962163],
        [3613599199827547, 14952528385270180253],
        [5364527230714204621, 7310770326300738069],
        [9246037983701160127, 3324605599585165047],
        [12728118561602933520, 6113373920773219421],
        [15371653297656955559, 7789645825304450812],
    ],
    [
        [4516510775002426513, 3566584478581873212],
        [8612959344442446869, 15360738315019883086],
        [4301249497429376384, 16418099929799381752],
        [15518596108656199155, 14633418594576613692],
        [8178806049610950682, 3256339422672622302],
        [756548248940552923, 17317528353089304043],
    ],
    [
        [15454817700272548110, 17198234262137136254],
        [2558361820929487520, 8601549678800449189],
        [12279306716889220222, 6504525294360065887],
        [4191690648182441120, 545979542586057044],
        [2924813836407529485, 16062886553258023468],
        [14294514175163210354, 2218325185229973763],
    ],
    [
        [10983372142071226281, 10673533601225795291],
        [16390764854169517766, 14318866971343881430],
        [9374687126228831323, 3888617263348830277],
        [1585852809540529411, 3645998864687605515],
        [3822950602532403092, 2945042110248410024],
        [3245700698079980596, 6863201950048280129],
    ],
    [
        [14804379758688002122, 2293284833968073399],
        [11882020160970452760, 2630479938823740919],
        [17146932215141377384, 4102507967983146841],
        [15791179214244650165, 17782256886776188725],
        [8818869438925288308, 542409447355246037],
        [5130643318760237204, 6135300259749075935],
    ],
    [
        [6667527891045103, 6051811410088945928],
        [13826038348507632762, 9333025458754159937],
        [10349531358254164720, 13818102641201195341],
        [11228575050680504825, 14171604677665562948],
        [408802311195972735, 13110537548489213391],
        [3175062453089130541, 4035664867916633890],
    ],
    [
        [6676544636053560174, 5393393270840365985],
        [8538520863276656999, 676989006079319585],
        [14516408521828760150, 16420262633366931380],
        [1299683309589890348, 10426742246916916461],
        [14459310219049212575, 4034707230021349394],
        [7853242450810480526, 18197298539884414108],
    ],
    [
        [9542991615947575929, 292061666645761691],
        [1991828379247330859, 15959081902805162035],
        [11445015640477324820, 5568719595272022292],
        [2464843717007188295, 10802309296363304108],
        [18253568032069809070, 18355530924964178033],
        [13029672139954537622, 13859789774884353662],
    ],
    [
        [15847630075660619709, 6083784184243903025],
        [7729982226222894712, 6823463114495331512],
        [14186611509182015525, 2431787156912765329],
        [16953679319920802798, 8245302757822828583],
        [13944170238824674708, 16369954344922594596],
        [7905833579457642655, 16368093408679101966],
    ],
    [
        [9129461369975929703, 13766441918594711256],
        [5568893035177306456, 4605528160275031226],
        [1551815709305302165, 4553365954250863656],
        [15787365519111812306, 3389064139401109858],
        [10742570171564288524, 9448120286568228713],
        [11184095349927048866, 13410493938165595350],
    ],
    [
        [8942808504924989119, 12576519392202999262],
        [7894648412541494164, 8921762426363813187],
        [16648000779790610434, 6223506581933215177],
        [13884390006094644263, 10069079183153993331],
        [135383389425047939, 14836084429363705369],
        [11989518386415735843, 6899566154613375523],
    ],
    [
        [2939790975933414624, 12147524706367442806],
        [7426467823017601018, 9375879036623728719],
        [12852011607736313476, 8357340672838956964],
        [10941106345851844436, 17850259290882241732],
        [7019005671728452146, 16797691642402486471],
        [13160621092620939014, 10432660606826335532],
    ],
];
const CASTLE_KEYS: [u64; 4] = [
    13903699408998609807,
    1394504070964353662,
    2001174967191055984,
    13016778750842461853,
];
const EP_KEYS: [u64; 8] = [
    3094884295509202524,
    661707550665752957,
    12428307272981688398,
    5790875449229408899,
    18108816876033594148,
    4586870879357451718,
    15226297474735306250,
    12936367968696784083,
];
pub const BLACK_TO_MOVE_KEY: u64 = 11183034114380226606;
