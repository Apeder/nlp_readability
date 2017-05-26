

```python
%run /Users/admin/pycommon
```

# Data Sampling


```python
# Read in 5 genre dataset
all_data_raw = sqlContext.sql("SELECT * FROM genre_NLP_comparison")
```


```python
all_data_raw.printSchema()
```

# Feature Generation

## Filter out posts too short to measure readability


```python
# Measure article length with number of sentences and number of words to filter dataset for Dale-Chall readability scoring. 

import pyspark
from pyspark.sql.functions import udf
import nltk
from nltk.data import load

nltk.download('punkt')

tokenizer = load('tokenizers/punkt/english.pickle')
                 
def gustavstokenizer(x):
    t = tokenizer.tokenize(x)
    return len(t) 

nsent = udf(lambda i: gustavstokenizer(i), pyspark.sql.types.IntegerType())

dfone = all_data_raw.withColumn('num_sent', nsent(all_data_raw.text))
```


```python
# Tokenize and count tokens to generate word count 

from nltk.tokenize.treebank import TreebankWordTokenizer
_treebank_word_tokenize = TreebankWordTokenizer().tokenize

def numwords(x):
  z = [token for sent in tokenizer.tokenize(x)
            for token in _treebank_word_tokenize(sent)]
  return len(z)

nwords = udf(lambda i: numwords(i), pyspark.sql.types.IntegerType())

dftwo = dfone.withColumn('num_words', nwords(dfone.text))
```


```python
# Filter text with fewer than 20 words in order to avoid articles with "0" sentences. Readability and Sentiment require dividing by sentence count, so this value must be >0. For some reason, this will work in small batches but will throw an error when trying to write a table or run a SQL query across a temporary table. 
# filtered = dftwo[dftwo.num_words > 20 & dftwo.num_sent > 1]
fil = dftwo.filter(dftwo.num_words > 20)
filtered = fil.filter(fil.num_sent > 2)
```

## Generate Dale-Chall Readability scores


```python
import string
exclude = list(set(string.punctuation))
easy_words = """a able aboard about above absent accept accident account ache aching acorn acre across act acts add address admire adventure afar	afraid after afternoon afterward afterwards again against age aged ago agree ah ahead aid aim air airfield airplane airport airship airy	alarm alike alive all alley alligator allow almost alone along aloud already also always am America American among amount an and	angel anger angry animal another answer ant any anybody anyhow anyone anything anyway anywhere apart apartment ape apiece appear apple April	apron are aren't arise arithmetic arm armful army arose around arrange arrive arrived arrow art artist as ash ashes aside ask	asleep at ate attack attend attention August aunt author auto automobile autumn avenue awake awaken away awful awfully awhile ax axe baa babe babies back background backward backwards bacon bad badge badly bag bake baker bakery baking ball balloon banana band bandage bang banjo bank banker bar barber bare barefoot barely bark barn barrel base baseball basement basket bat batch bath bathe bathing bathroom bathtub	battle battleship bay be beach bead beam bean bear beard beast beat beating beautiful beautify beauty became because become becoming bed bedbug bedroom bedspread bedtime bee beech beef beefsteak beehive been beer beet before beg began beggar begged begin beginning begun behave behind being	believe bell belong below belt bench bend beneath bent berries berry beside besides best bet better between bib bible bicycle bid big bigger bill billboard bin bind bird birth birthday biscuit bit bite biting bitter black blackberry blackbird blackboard blackness blacksmith blame blank blanket	blast blaze bleed bless blessing blew blind blindfold blinds block blood bloom blossom blot blow blue blueberry bluebird blush board boast boat bob bobwhite bodies body boil boiler bold bone bonnet boo book bookcase bookkeeper boom boot born borrow boss both bother bottle bottom	bought bounce bow bowl bow-wow box boxcar boxer boxes boy boyhood bracelet brain brake bran branch brass brave bread break breakfast breast breath breathe breeze brick bride bridge bright brightness bring broad broadcast broke broken brook broom brother brought brown brush bubble bucket buckle	bud buffalo bug buggy build building built bulb bull bullet bum bumblebee bump bun bunch bundle bunny burn burst bury bus bush bushel business busy but butcher butt butter buttercup butterfly buttermilk butterscotch button buttonhole buy buzz by bye cab cabbage cabin cabinet cackle cage cake calendar calf call caller calling came camel camp campfire can canal canary candle candlestick candy cane cannon cannot canoe can't canyon cap cape capital captain car card cardboard care careful careless carelessness carload carpenter carpet carriage carrot carry cart	carve case cash cashier castle cat catbird catch catcher caterpillar catfish catsup cattle caught cause cave ceiling cell cellar cent center cereal certain certainly chain chair chalk champion chance change chap charge charm chart chase chatter cheap cheat check checkers cheek cheer cheese cherry chest chew	chick chicken chief child childhood children chill chilly chimney chin china chip chipmunk chocolate choice choose chop chorus chose chosen christen Christmas church churn cigarette circle circus citizen city clang clap class classmate classroom claw clay clean cleaner clear clerk clever click cliff climb clip cloak	clock close closet cloth clothes clothing cloud cloudy clover clown club cluck clump coach coal coast coat cob cobbler cocoa coconut cocoon cod codfish coffee coffeepot coin cold collar college color colored colt column comb come comfort comic coming company compare conductor cone connect coo cook	cooked cooking cookie cookies cool cooler coop copper copy cord cork corn corner correct cost cot cottage cotton couch cough could couldn't count counter country county course court cousin cover cow coward cowardly cowboy cozy crab crack cracker cradle cramps cranberry crank cranky crash crawl crazy	cream creamy creek creep crept cried croak crook crooked crop cross crossing cross-eyed crow crowd crowded crown cruel crumb crumble crush crust cry cries cub cuff cup cuff cup cupboard cupful cure curl curly curtain curve cushion custard customer cut cute cutting dab dad daddy daily dairy daisy dam damage dame damp dance dancer dancing dandy danger dangerous dare dark darkness darling darn dart dash date daughter	dawn day daybreak daytime dead deaf deal dear death December decide deck deed deep deer defeat defend defense delight den dentist depend deposit describe desert	deserve desire desk destroy devil dew diamond did didn't die died dies difference different dig dim dime dine ding-dong dinner dip direct direction dirt dirty	discover dish dislike dismiss ditch dive diver divide do dock doctor does doesn't dog doll dollar dolly done donkey don't door doorbell doorknob doorstep dope	dot double dough dove down downstairs downtown dozen drag drain drank draw drawer draw drawing dream dress dresser dressmaker drew dried drift drill drink drip	drive driven driver drop drove drown drowsy drub drum drunk dry duck due dug dull dumb dump during dust dusty duty dwarf dwell dwelt dying each eager eagle ear early earn earth east eastern easy eat eaten	edge egg eh eight eighteen eighth eighty either elbow elder eldest electric	electricity elephant eleven elf elm else elsewhere empty end ending enemy engine	engineer English enjoy enough enter envelope equal erase eraser errand escape eve	even evening ever every everybody everyday everyone everything everywhere evil exact except	exchange excited exciting excuse exit expect explain extra eye eyebrow fable face facing fact factory fail faint fair fairy faith fake fall false family fan fancy far faraway fare farmer farm farming far-off farther fashion fast fasten fat father	fault favor favorite fear feast feather February fed feed feel feet fell fellow felt fence fever few fib fiddle field fife fifteen fifth fifty fig fight figure file fill	film finally find fine finger finish fire firearm firecracker fireplace fireworks firing first fish fisherman fist fit fits five fix flag flake flame flap flash flashlight flat flea flesh	flew flies flight flip flip-flop float flock flood floor flop flour flow flower flowery flutter fly foam fog foggy fold folks follow following fond food fool foolish foot football	footprint for forehead forest forget forgive forgot forgotten fork form fort forth fortune forty forward fought found fountain four fourteen fourth fox frame free freedom freeze freight French fresh	fret Friday fried friend friendly friendship frighten frog from front frost frown froze fruit fry fudge fuel full fully fun funny fur furniture further fuzzy gain gallon gallop game gang garage garbage garden gas gasoline gate gather gave gay gear geese general gentle gentleman gentlemen	geography get getting giant gift gingerbread girl give given giving glad gladly glance glass glasses gleam glide glory glove glow	glue go going goes goal goat gobble God god godmother gold golden goldfish golf gone good goods goodbye good-by goodbye	good-bye good-looking goodness goody goose gooseberry got govern government gown grab gracious grade grain grand grandchild grandchildren granddaughter grandfather grandma	grandmother grandpa grandson grandstand grape grapes grapefruit grass grasshopper grateful grave gravel graveyard gravy gray graze grease great green greet	grew grind groan grocery ground group grove grow guard guess guest guide gulf gum gun gunpowder guy ha habit had hadn't hail hair haircut hairpin half hall halt ham hammer hand handful handkerchief handle handwriting hang happen happily happiness happy harbor hard hardly hardship hardware hare hark	harm harness harp harvest has hasn't haste hasten hasty hat hatch hatchet hate haul have haven't having hawk hay hayfield haystack he head headache heal health healthy heap hear hearing	heard heart heat heater heaven heavy he'd heel height held hell he'll hello helmet help helper helpful hem hen henhouse her hers herd here here's hero herself he's hey hickory	hid hidden hide high highway hill hillside hilltop hilly him himself hind hint hip hire his hiss history hit hitch hive ho hoe hog hold holder hole holiday hollow holy	home homely homesick honest honey honeybee honeymoon honk honor hood hoof hook hoop hop hope hopeful hopeless horn horse horseback horseshoe hose hospital host hot hotel hound hour house housetop	housewife housework how however howl hug huge hum humble hump hundred hung hunger hungry hunk hunt hunter hurrah hurried hurry hurt husband hush hut hymn I ice icy I'd idea ideal if ill	I'll I'm important impossible improve in inch inches	income indeed Indian indoors ink inn insect inside	instant instead insult intend interested interesting into invite	iron is island isn't it its it's itself	I've ivory ivy jacket jacks jail jam January jar	jaw jay jelly jellyfish jerk jig	job jockey join joke joking jolly	journey joy joyful joyous judge jug	juice juicy July jump June junior	junk just keen keep kept kettle key	kick kid kill killed kind	kindly kindness king kingdom kiss	kitchen kite kitten kitty knee	kneel knew knife knit knives	knob knock knot know known lace lad ladder ladies lady laid lake lamb lame lamp land lane language lantern lap lard large lash lass last	late laugh laundry law lawn lawyer lay lazy lead leader leaf leak lean leap learn learned least leather leave leaving	led left leg lemon lemonade lend length less lesson let let's letter letting lettuce level liberty library lice lick lid	lie life lift light lightness lightning like likely liking lily limb lime limp line linen lion lip list listen lit	little live lives lively liver living lizard load loaf loan loaves lock locomotive log lone lonely lonesome long look lookout	loop loose lord lose loser loss lost lot loud love lovely lover low luck lucky lumber lump lunch lying machine machinery mad made magazine magic maid mail mailbox mailman major make making male mama mamma man manager mane manger many map	maple marble march March mare mark market marriage married marry mask mast master mat match matter mattress may May maybe mayor maypole me	meadow meal mean means meant measure meat medicine meet meeting melt member men mend meow merry mess message met metal mew mice middle	midnight might mighty mile milk milkman mill miler million mind mine miner mint minute mirror mischief miss Miss misspell mistake misty mitt mitten	mix moment Monday money monkey month moo moon moonlight moose mop more morning morrow moss most mostly mother motor mount mountain mouse mouth	move movie movies moving mow Mr. Mrs. much mud muddy mug mule multiply murder music must my myself nail name nap napkin narrow nasty naughty navy near nearby	nearly neat neck necktie need needle needn't Negro neighbor neighborhood	neither nerve nest net never nevermore new news newspaper next	nibble nice nickel night nightgown nine nineteen ninety no nobody	nod noise noisy none noon nor north northern nose not	note nothing notice November now nowhere number nurse nut oak oar oatmeal oats obey ocean o'clock October odd of off	offer office officer often oh oil old old-fashioned on once one	onion only onward open or orange orchard order ore organ other	otherwise ouch ought our ours ourselves out outdoors outfit outlaw outline	outside outward oven over overalls overcoat overeat overhead overhear overnight overturn	owe owing owl own owner ox pa pace pack package pad page paid pail pain painful paint painter painting pair pal palace pale pan pancake pane pansy pants papa paper parade pardon parent park part partly partner party	pass passenger past paste pasture pat patch path patter pave pavement paw pay payment pea peas peace peaceful peach peaches peak peanut pear pearl peck peek peel peep peg pen pencil penny	people pepper peppermint perfume perhaps person pet phone piano pick pickle picnic picture pie piece pig pigeon piggy pile pill pillow pin pine pineapple pink pint pipe pistol pit pitch pitcher pity	place plain plan plane plant plate platform platter play player playground playhouse playmate plaything pleasant please pleasure plenty plow plug plum pocket pocketbook poem point poison poke pole police policeman polish polite	pond ponies pony pool poor pop popcorn popped porch pork possible post postage postman pot potato potatoes pound pour powder power powerful praise pray prayer prepare present pretty price prick prince princess	print prison prize promise proper protect proud prove prune public puddle puff pull pump pumpkin punch punish pup pupil puppy pure purple purse push puss pussy pussycat put putting puzzle quack quart	quarter queen	queer question	quick quickly	quiet quilt	quit quite rabbit race rack radio radish rag rail railroad railway rain rainy rainbow raise raisin rake ram ran ranch rang rap rapidly	rat rate rather rattle raw ray reach read reader reading ready real really reap rear reason rebuild receive recess record red	redbird redbreast refuse reindeer rejoice remain remember remind remove rent repair repay repeat report rest return review reward rib ribbon rice	rich rid riddle ride rider riding right rim ring rip ripe rise rising river road roadside roar roast rob robber robe	robin rock rocky rocket rode roll roller roof room rooster root rope rose rosebud rot rotten rough round route row rowboat	royal rub rubbed rubber rubbish rug rule ruler rumble run rung runner running rush rust rusty rye sack sad saddle sadness safe safety said sail sailboat sailor saint salad sale salt same sand sandy sandwich sang sank sap sash sat satin satisfactory Saturday sausage savage save savings saw say scab scales scare scarf school schoolboy schoolhouse schoolmaster schoolroom scorch score scrap scrape scratch scream screen screw scrub sea seal seam search season seat second secret see seeing seed seek seem seen seesaw select self selfish	sell send sense sent sentence separate September servant serve service set setting settle settlement seven seventeen seventh seventy several sew shade shadow shady shake shaker shaking shall shame shan't shape share sharp shave she she'd she'll she's shear shears shed sheep sheet shelf shell shepherd shine shining shiny ship shirt shock shoe shoemaker shone shook shoot shop shopping shore short shot should shoulder shouldn't shout shovel show shower	shut shy sick sickness side sidewalk sideways sigh sight sign silence silent silk sill silly silver simple sin since sing singer single sink sip sir sis sissy sister sit sitting six sixteen sixth sixty size skate skater ski skin skip skirt sky slam slap slate slave sled sleep sleepy sleeve sleigh slept slice slid slide sling slip slipped slipper slippery slit slow slowly sly smack small smart smell	smile smoke smooth snail snake snap snapping sneeze snow snowy snowball snowflake snuff snug so soak soap sob socks sod soda sofa soft soil sold soldier sole some somebody somehow someone something sometime sometimes somewhere son song soon sore sorrow sorry sort soul sound soup sour south southern space spade spank sparrow speak speaker spear speech speed spell spelling spend spent spider spike spill spin spinach spirit spit	splash spoil spoke spook spoon sport spot spread spring springtime sprinkle square squash squeak squeeze squirrel stable stack stage stair stall stamp stand star stare start starve state station stay steak steal steam steamboat steamer steel steep steeple steer stem step stepping stick sticky stiff still stillness sting stir stitch stock stocking stole stone stood stool stoop stop stopped stopping store stork stories storm stormy story stove straight	strange stranger strap straw strawberry stream street stretch string strip stripes strong stuck study stuff stump stung subject such suck sudden suffer sugar suit sum summer sun Sunday sunflower sung sunk sunlight sunny sunrise sunset sunshine supper suppose sure surely surface surprise swallow swam swamp swan swat swear sweat sweater sweep sweet sweetness sweetheart swell swept swift swim swimming swing switch sword swore table tablecloth tablespoon tablet tack tag tail tailor take taken taking tale talk talker tall tame tan tank tap tape tar tardy task taste taught tax tea teach teacher team tear	tease teaspoon teeth telephone tell temper ten tennis tent term terrible test than thank thanks thankful Thanksgiving that that's the theater thee their them then there these they they'd they'll they're	they've thick thief thimble thin thing think third thirsty thirteen thirty this thorn those though thought thousand thread three threw throat throne through throw thrown thumb thunder Thursday thy tick ticket	tickle tie tiger tight till time tin tinkle tiny tip tiptoe tire tired title to toad toadstool toast tobacco today toe together toilet told tomato tomorrow ton tone tongue tonight too	took tool toot tooth toothbrush toothpick top tore torn toss touch tow toward towards towel tower town toy trace track trade train tramp trap tray treasure treat tree trick tricycle tried	trim trip trolley trouble truck true truly trunk trust truth try tub Tuesday tug tulip tumble tune tunnel turkey turn turtle twelve twenty twice twig twin two ugly umbrella uncle under understand underwear	undress unfair unfinished unfold unfriendly unhappy	unhurt uniform United States unkind unknown	unless unpleasant until unwilling up upon	upper upset upside upstairs uptown upward	us use used useful valentine valley valuable	value vase vegetable	velvet very vessel	victory view village	vine violet visit	visitor voice vote wag wagon waist wait wake waken walk wall walnut want war warm warn was wash washer washtub wasn't waste watch watchman water watermelon waterproof wave wax	way wayside we weak weakness weaken wealth weapon wear weary weather weave web we'd wedding Wednesday wee weed week we'll weep weigh welcome well went were	we're west western wet we've whale what what's wheat wheel when whenever where which while whip whipped whirl whisky whiskey whisper whistle white who who'd whole	who'll whom who's whose why wicked wide wife wiggle wild wildcat will willing willow win wind windy windmill window wine wing wink winner winter wipe wire	wise wish wit witch with without woke wolf woman women won wonder wonderful won't wood wooden woodpecker woods wool woolen word wore work worker workman world	worm worn worry worse worst worth would wouldn't wound wove wrap wrapped wreck wren wring write writing written wrong wrote wrung yard yarn year yell	yellow yes yesterday yet	yolk yonder you you'd	you'll young youngster your	yours you're yourself yourselves	youth you've"""
easy_word_set = set(easy_words.split())
```


```python
import re
#  None values being produced from Textstat module, though a lighlty modified version of the code below runs. All code below opensource from https://github.com/shivam5992/textstat/blob/master/textstat/textstat.py. However, this produced scores that were suspiciously low, so the original method was used after some additional troubleshooting. 
def sentence_count(text):
		ignoreCount = 0
		sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
		for sentence in sentences:
			if lexicon_count(sentence) <= 2:
				ignoreCount = ignoreCount + 1
		return max(1, len(sentences) - ignoreCount)

def lexicon_count(text, removepunct=True):
		if removepunct:
			text = ''.join(ch for ch in text if ch not in exclude)
		count = len(text.split())
		return count
      
def syllable_count(text):
  count = 0
  vowels = 'aeiouy'
  text = text.lower()
  text = "".join(x for x in text if x not in exclude)
  
  if text == None:
    return 0
  elif len(text) == 0:
			return 0
  else:
    if text[0] in vowels:
      count += 1
    for index in range(1, len(text)):
          if text[index] in vowels and text[index-1] not in vowels:
            count += 1
          if text.endswith('e'):
            count -= 1
          if text.endswith('le'):
            count += 1
          if count == 0:
            count += 1
          count = count - (0.1*count)
          return count      
      
def avg_sentence_length(text):
		lc = lexicon_count(text)
		sc = sentence_count(text)
		try:
			ASL = float(lc/sc)
			return round(lc/sc, 1)
		except:
			print("Error(ASL): Sentence Count is Zero, Cannot Divide")
			return
      
def diff_words(text):
	text_list = text.split()
	diff_words_set = set()
	for value in text_list:
		if value not in easy_word_set:
			if syllable_count(value) > 1:
				if value not in diff_words_set:
					diff_words_set.add(value)
	return len(diff_words_set)

def dale_chall_readability_score(text):
  word_count = lexicon_count(text)
  diff = diff_words(text)
  count = word_count-diff
  if word_count > 0:
      per = float(count)/float(word_count)*100
  else:
	print("Error(DCRS): Word Count is zero cannot divide")
	return 
  difficult_words = 100-per
  if difficult_words > 5:
    score = (0.1579 * difficult_words) + (0.0496 * avg_sentence_length(text)) + 3.6365
  else:
    score = (0.1579 * difficult_words) + (0.0496 * avg_sentence_length(text))
  return round(score, 2)
```


```python
def rdl(j):
  l = dale_chall_readability_score(j)
  return l

# Still returning some none values
rdL = udf(lambda x: rdl(x), pyspark.sql.types.DoubleType())

readability = filtered.withColumn('Readability', rdL(filtered.text))
```


```python
#This produces higher scores than above, though now does not return none values.  Seems more likely that articles are in a higher range than a lower one. 
from textstat.textstat import textstat

def readingL(j):
  l = textstat.dale_chall_readability_score(j)
  return l


readL = udf(lambda x: readingL(x), pyspark.sql.types.DoubleType())

readability = filtered.withColumn('Readability', readL(filtered.text))
```


```python
readability.printSchema()
```


```python
readability.show(5)
```


```python
# Writing managed tables seems to be time-intensive and error prone when datasets are too large.  Possibly more efficient to store Medium's text corpus as .txt or .csv files using dbfsutil?

#Writing only 4 four columns eliminates errors. Runs in a little over 20 min. 
readability_scores = readability.select('post_ID', 'Readability', 'num_words', 'num_sent')
readability_scores.write.saveAsTable("readability_scores", mode='overwrite')

# dbutils.fs.mkdirs("/genre_NLP_comparison_raw_text")
# raw_text.saveAsTextFile("/genre_NLP_comarison_raw_text/text_samples.txt")
# Oct_Jan_politics_dip_text.saveAsTextFile("/genre_NLP_comparison_raw_text/politics_samples.txt")

# An error occurred while calling o384.saveAsTabl, Caused by: org.apache.spark.SparkException: Job aborted due to stage failure: Task 3 in stage 151932.0 failed 4 times, most recent failure: Lost task 3.3 in stage 151932.0 (TID 129164, ip-10-50-237-52.us-west-2.compute.internal): java.nio.channels.ClosedChannelException
```


```python
ll = sqlContext.sql("SELECT * FROM readability_scores")
ll.show(5)
```


```python
ll.describe('Readability').show()
```

# Visualization and Analysis


```python
%r
Readability_genre_comp <- collect(sql(sqlContext, "SELECT g.post_ID, date, num_words, ln(num_words) AS log_num_wrds, ln(total_ttr) AS log_ttr, Readability, ln(Readability) AS log_read, ln(drafting_time) AS log_draft_time, genre AS Genre FROM readability_scores AS r JOIN genre_NLP_comparison AS g ON r.post_ID = g.post_ID WHERE date < '2016-05-23'"))
```


```python
%r
str(Readability_genre_comp)
```


```python
%r 
summary(Readability_genre_comp)
```


```python
%r
sd(Readability_genre_comp$Readability)
```

## Sample Random examples of posts at varied reading levels


```python
%r
print(Readability_genre_comp[Readability_genre_comp$Readability==27.46,])
```


```python
%r
print(Readability_genre_comp[Readability_genre_comp$Readability==.2,])
```


```python
%r
low_read <- collect(sql(sqlContext, "SELECT post_ID, Readability FROM readability_scores WHERE num_words > 400 AND Readability BETWEEN 5 AND 7 ORDER BY Readability ASC"))
mid_read <- collect(sql(sqlContext, "SELECT post_ID, Readability FROM readability_scores WHERE num_words > 400 AND Readability BETWEEN 7 AND 9 ORDER BY Readability ASC"))
high_read <- collect(sql(sqlContext, "SELECT post_ID, Readability FROM readability_scores WHERE num_words > 400 AND Readability BETWEEN 9 AND 11 ORDER BY Readability ASC"))
```


```python
%r
print(head(low_read))

print(head(mid_read))

print(head(high_read))
print(tail(high_read))
```


```python
%r
six <- low_read[low_read$Readability ==6,]
print(head(six))
```


```python
%r
eight <- mid_read[mid_read$Readability==8,]
print(head(eight))
```


```python
%r
ten <- high_read[high_read$Readability==10,]
print(head(ten))
```

## Visualize


```python
%r
#Possible to label other y axis with grade levels? 
cl <- na.omit(Readability_genre_comp)
cl <- cl[cl$Readability < 13 & cl$Readability > 4,]

plott <- ggplot(cl, aes(x=Genre, y=Readability, color=Genre)) + 
  scale_y_continuous(breaks=seq(0,30,.25)) +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75), aes(group=Genre)) + 
  geom_hline(yintercept=6, colour='springgreen4') + 
  geom_hline(yintercept=median(cl$Readability), color='red') +
  theme(legend.position="none") + 
  labs(title="Readability Score Distributions, October 2015 to April 2016", y="Dale-Chall Readability")

plott
```


```python
%r
cl <- na.omit(Readability_genre_comp)
plotty <- ggplot(cl, aes(x=genre, y=Readability, color=genre)) + 
  geom_boxplot(aes(group=genre), outlier.colour = "red", outlier.shape = 1)

plotty

# ggplot(diamonds, aes(carat, price)) +
#   geom_boxplot(aes(group = cut_width(carat, 0.25)))
```


```python
%r
library(ggplot2)

Readability_genre_comp$genre <- as.factor(Readability_genre_comp$genre)

ggplot(Readability_genre_comp, aes(Readability, color=genre)) + 
  geom_density() + 
  xlim(4.5,13) + 
  geom_vline(xintercept=6, colour='springgreen4') + 
  geom_vline(xintercept=mean(Readability_genre_comp$Readability), color='red')
```


```python
%r
u = Readability_genre_comp[Readability_genre_comp$genre == 'Education',]
head(u)
```


```python
%r
library(ggplot2)

# tp_scale <- subset(Readability_genre_comp, Readability_genre_comp$total_ttr > 0 & Readability_genre_comp$drafting_time > 0)

tp_scale <- Readability_genre_comp

# ttr_read <- Readability_genre_comp[which(Readability_genre_comp$log_read > -1 & Readability_genre_comp$log_read < 1), ] 
# ttr_read$log_read <- ttr_read$log_read + abs(min(ttr_read$log_read))

# ttr <- ggplot(ttr_read, aes(y = ttr_read$log_ttr, x = ttr_read$log_read, color = genre)) +
#   geom_point(alpha=.1, color='springgreen4') + 
#   geom_smooth()     

ttr_orig <- ggplot(tp_scale, aes(y=tp_scale$log_ttr, x=tp_scale$Readability, color=genre))+
#   geom_point(alpha=.1, color='blue') +
  geom_smooth(se=FALSE) + 
  xlim(5,12)

tp_scale <- Readability_genre_comp
labs(title="Sentiment and Reading Time, October 2015 to April 2016", x="Sentiment", y="Total Time Read (TTR)")

ttr_read_final <- ggplot(tp_scale, aes(y=tp_scale$log_ttr, x=tp_scale$log_read, color=genre))+
#   geom_point(alpha=.1, color='blue') +
  geom_smooth(se=FALSE) + 
  xlim(1.6,2.5)

dt <- ggplot(tp_scale, aes(y = tp_scale$log_read, x = tp_scale$log_draft_time, color=genre))+
  geom_point(alpha=.1, color='springgreen4') + 
  geom_smooth() +
  ylim(1.5, 3)
```


```python
%r
dt

#No meaningful relationship between time spent drafting and readability.j 
```


```python
%r
model <- lm(log_read~log_draft_time, data=tp_scale, family="gaussian")
summary(model)
```


```python
%r
ttr_read_final
```


```python
%r
tp_scale <- Readability_genre_comp

ttr_orig <- ggplot(tp_scale, aes(y=tp_scale$log_ttr, x=tp_scale$Readability, color=Genre))+
  scale_x_continuous(limits=c(5,12), breaks=c(5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12)) +
  scale_y_continuous(breaks=seq(11,15,.25)) +
#   geom_point(alpha=.1, color='blue') +
  geom_smooth(se=FALSE) + 
#   xlim(5,12) + 
  labs(title="Readability and Reading Time, October 2015 to April 2016", x="Dale-Chall Readability", y="Total Time Read (TTR), Natural Log Scale")

ttr_orig
```


```python
%r
#Natural log comparison coefficients are about the same as percentage change, so for each unit increase in readability, ttr decreases by 40%. 
model <- lm(log_ttr~log_read, data=tp_scale, family="gaussian")
summary(model)
```


```python
%r
#When we control for the length of the articles, readabilities coefficient becomes positive. This could be multicolinearity, but VIF was only 1. For each unit increase in readability, ttr increases by nearly 75%. 
model <- lm(log_ttr~log_read+log(num_words)+log_draft_time, data=tp_scale, family="gaussian")
summary(model)
```


```python
%r
model <- glm(log_ttr~log_read, data=ttr_read, family="gaussian")
summary(model)
```

# Initial Exploratory Plots


```python
%r
# Only political articles
TestPlot <- read.parquet(sqlContext, "readabilitytest.parquet")
```


```python
%r
# Since readability and ttr show a weak relationship, other variables probably influence ttr more than readability. Try to compare readability and drafting time? 

library(ggplot2)

old_read <- na.omit(as.data.frame(TestPlot))
# old_read$total_ttr <- old_read$total_ttr[old_read$total_ttr > 0,]
# tp_scale <- na.omit(tp_scale)

ts <- ggplot(old_read, aes(y = log(old_read$total_ttr), x = log(old_read$Readability))) +
  geom_point(alpha=.1, color='springgreen4') + 
  geom_smooth() +
  xlim(1.5,3) + ylim(5, 22)
ts
```


```python
%r
summary(old_read)
```


```python
%r
str(old_read)
```


```python
%r 
old_read$log_ttr <- log(old_read$total_ttr)
old_read$log_read <- log(old_read$Readability)

old_read_clean <- na.omit(old_read)

old_read_clean$log_ttr[which(is.nan(old_read_clean$log_ttr))] = NA
old_read_clean$log_ttr[which(old_read_clean$log_ttr==Inf)] = NA

old_read_clean$log_read[which(is.nan(old_read_clean$log_read))] = NA
old_read_clean$log_read[which(old_read_clean$log_read==Inf)] = NA
```


```python
%r

ln <- ggplot(tp_scale, aes(y = tp_scale$log_ttr, x = tp_scale$log_num_wrds, color=genre))+
  geom_point(alpha=.1, color='springgreen4') + 
  geom_smooth()  
#   xlim(0,3000) + ylim(5, 25)

tp_clean <- na.omit(tp_scale)

mn <- ggplot(tp_clean, aes(y=tp_clean$log_read, x=tp_clean$log_num_wrds, color=genre)) + 
  geom_point(alpha=.1, color='blue') + 
  geom_smooth()  
#   xlim(0,5000) + ylim(0,5)
```


```python
%r
summary(tp_clean)
```


```python
%r
ln
```


```python
%r
model <- lm(log_ttr~log_num_wrds, data=tp_scale, family="gaussian")
summary(model)
```


```python
%r
mn
```


```python
%r
#Slightly correlated, but not enough to skew the results or ruin the model.
model <- lm(log_read~log_num_wrds, data=tp_clean, family="gaussian")
summary(model)
```


```python
%r
plot(model)
```


```python
%r
#Relationship between TTR and readability looks sinusoidal. Easy to read articles seem to get increased ttr, which then falls, before peaking, falling and beginning to rise again. Except for education, which only has one peak. 

time <- ggplot(tp_scale, aes(y = tp_scale$log_ttr, x = tp_scale$log_read, color = genre)) +
#   geom_point(alpha=.1, color='springgreen4') + 
  geom_smooth(se=FALSE) +  
  xlim(0,3) #+ 
#   facet_grid(.~genre)

time
```


```python
%r
help(log)
```


```python
%r
# Readability becomes a significant coefficient for TTR after drafting time is accounted for in the model; however, R-squared only increases slightly
model <- lm(log_ttr~log_read + log_draft_time, data=tp_scale_cpy, family="gaussian")
summary(model)
```


```python
%r

# Adding log_num_wrds to the model changes log_read's coefficient to negative. http://stats.stackexchange.com/questions/1580/regression-coefficients-that-flip-sign-after-including-other-predictors Multicollinearity en.wikipedia.org/wiki/Multicollinearity causing this? No, this is a strange example of Simpson's Paradox: https://en.wikipedia.org/wiki/Simpson%27s_paradox.  Possibly controlling for article length via number of words gives us a clearer view of the relationship between ttr and readability? Article length via number of words is a "lurking confounder"? R^2 is relatively low, at .3552. 

#Looks like article length (number of words) is a suppressor variable for Readability. Need to control for article length to get an accurate regression. http://home.ubalt.edu/tmitch/645/articles/Thompson%20%26%20Levine%20Ex%20supressor%20vars.pdf

#How to convert these log based coefficients into the orignal units? 
model_penult <- glm(log_ttr~log_draft_time + log_num_wrds + sin(log_read) + cos(log_read), data=tp_scale)
summary(model_penult)
```


```python
%r
plot(model_penult)
```


```python
%r
t <- ggplot(tp_clean, aes(y=tp_clean$log_ttr, x = tp_clean$log_read)
abline(model_penult)
```


```python
%r
model_penult <- glm(log_ttr~log_draft_time + log_num_wrds + cos(log_read), data=tp_scale)
summary(model_penult)
```


```python
%r
monely <- lm(log_ttr~log_draft_time + log_num_wrds + log_read, data=tp_scale)
summary(monely)
```


```python
%r
#VIF checks multicollinearity 
# install.packages('usdm')
library(car)
# vif(tp_scale$log_read, tp_scale$log_draft_time) # Variance inflation factors (VIF) should be under 3 or 4

vif(model)
```


```python
%r
library(car)
vif(model_penult)
#Does not look like multicollinearity is a problem
```


```python
%r
help(vif)
```


```python
%r
# Residuals look ok, normally distributed
plot(density(resid(model_penult)))

```


```python
%r
summary(tp_scale)
```


```python
%r
#Residuals for number of words and readability looked skewed, so took the log
tp_clean <- na.omit(tp_scale)

par(mfrow=c(2,2))
plot(tp_clean$log_draft_time,residuals(model_penult))
plot(tp_clean$log_num_wrds,residuals(model_penult))
plot(tp_clean$log_read,residuals(model_penult))
```


```python
%r
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
```


```python
%r
multiplot(ttr, dt, ln, cols=1)
```


```python
%r

tp_clean <- na.omit(tp_scale)

rdblty <- ggplot(tp_scale, aes(x=tp_clean$genre, y=tp_clean$readability)) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width=.1, fill="black", outlier.colour=NA) + 
  stat_summary(fun.y=median, geom="point", fill="white", shape=21, size=2.5)

rdblty
```


```python
%r

# Check previous analysis of ttr and drafting time - Can't reproduce original results 
library(ggplot2)

all_data_raw <- collect(sql(sqlContext, "SELECT * FROM genre_NLP_comparison WHERE drafting_time > 0 AND total_ttr > 0"))

t <- ggplot(all_data_raw, aes(log(drafting_time), log(total_ttr))) +
  geom_point(alpha=.2, color='springgreen4') + 
  geom_smooth()

t
```


```python
%r 
model <- lm(log(total_ttr)~log(drafting_time), data=all_data_raw, family="gaussian")
summary(model)
```


```python
# %load_ext rpy2.ipython

```


```python
%%R

library(ggplot2)

ggplot(data.frame(x=c(0, 10)), aes(x)) + stat_function(fun=function(x) + sin(x) + log(x))
```


![png](/Users/andrewpederson/Portfolio/NLP_Readability/img/output_78_0.png)



```python
%%R
ggplot(data.frame(x=c(5, 12)), aes(x)) + stat_function(fun=function(x) + 3*sin(x) + log(x) + 15)
```


![png](/Users/andrewpederson/Portfolio/NLP_Readability/img/output_79_0.png)



```python
%%R
ggplot(data.frame(x=c(5, 12)), aes(x)) + stat_function(fun=function(x) + sin(x) + 13)
```


![png](/Users/andrewpederson/Portfolio/NLP_Readability/img/output_80_0.png)



```python
%%R
ggplot(data.frame(x=c(5, 12)), aes(x)) + stat_function(fun=function(x) + log(x)^-x)
```


![png](/Users/andrewpederson/Portfolio/NLP_Readability/img/output_81_0.png)



```python
%%R

# 
ggplot(data.frame(x=c(5, 12)), aes(x)) + stat_function(fun=function(x) + log(x)^-10*x *sin(x) + 13)
```


![png](/Users/andrewpederson/Portfolio/NLP_Readability/img/output_82_0.png)



```python
%%R
ggplot(data.frame(x=c(0, 20)), aes(x)) + stat_function(fun=function(x) + .2*sin(2*pi*.1*x) + 13)
    

```


![png](/Users/andrewpederson/Portfolio/NLP_Readability/img/output_83_0.png)



```python
%%R

# Damped sine wave with amplitude decay over time. 
ggplot(data.frame(x=c(0, 20)), aes(x)) + stat_function(fun=function(x) + (log(x)^-.2*x)*sin(.8*x-pi/2) + 13)
    
# y = 1 + sin(x - pi/3).*exp(-0.2*x)
    

```


![png](/Users/andrewpederson/Portfolio/NLP_Readability/img/output_84_0.png)



```python
%%R

# Damped sine wave with amplitude decay over time. 

#Amplitude coefficient
a <- -28
# Amplitude modulation
b <- 15
# Frequency modulation
c <- 16
# Y intercept
d <- 11.25
ggplot(data.frame(x=c(0, 100)), aes(x)) + stat_function(fun=function(x) + (a/(b+x))*cos((2*pi/c)*x) + d)
```


![png](/Users/andrewpederson/Portfolio/NLP_Readability/img/output_85_0.png)



```python

```
