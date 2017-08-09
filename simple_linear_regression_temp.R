# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set) # dependant variable ~ independant variable

# P value < 5% means the independent variable has significant effectssu

#Predict the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visual of Training and Test set observations
# install.packages('ggplot2')
#library(ggplot2)

#First plot all observations points , start with trainng set
ggplot() + 
  geom_point(aes (x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience,predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Traing set)') + 
  xlab('Years of experience') +
  ylab('Salary')

# Test set
ggplot() + 
  geom_point(aes (x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience,predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') + 
  xlab('Years of experience') +
  ylab('Salary')