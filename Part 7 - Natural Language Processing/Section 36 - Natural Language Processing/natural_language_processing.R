# Importing the dataset
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = "", stringsAsFactors = F)

# Clearning the texts
library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
# removendo espaços duplicados
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of World model
dtm = DocumentTermMatrix(corpus)
# mostra informações sobre a matrix
dtm
# manter somente 99.9% dos termos mais frequentes remove quase 700 termos
dtm = removeSparseTerms(dtm, 0.999)
dtm

# transforma a matrix dtm em um dataframe, para ser usado no mnodelo
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor
# decision tre, random forest r naive bayes precisam dessa linha depois trabalham com factores
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset in train set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Random Forest Classification

# Fitting Random Forest Classification to the dataset
library(randomForest)
cla = randomForest(x = train_set[-692],
                   y = train_set$Liked,
                   ntree = 10)
summary(cla)

# Feature scaling
# não precisa de feature scaling, pois só existem 0s e 1s
# train_set[,1:2] = scale(train_set[,1:2])
# test_set[,1:2]  = scale(test_set[,1:2])

# Predicting the Test set results
Y_pred = predict(cla, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set$Liked, Y_pred)
