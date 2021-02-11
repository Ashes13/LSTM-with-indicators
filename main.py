from MyLSTM import MyLSTM
import numpy as np
import pandas as pd

class MultidimensionalHorizontalFlange(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2021, 1, 12)  # Set Start Date
        self.SetCash(100000)  # Set Strategy Cash
        
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        
        self.SetExecution(ImmediateExecutionModel())

        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())

        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetUniverseSelection(LiquidETFUniverse())
        
        # Helper dictionaries
        self.macro_symbols = {'Bull' : Symbol.Create('AAPL', SecurityType.Equity, Market.USA)}
        self.models = {'Bull': None, 'Bear': None}

        # Use Train() method to avoid runtime error
        self.Train(self.TrainMyModel)
        #self.Train(self.DateRules.MonthEnd(), self.TimeRules.At(8,0), self.TrainMyModel)
        
        # Schedule prediction and plotting
        self.AddEquity('AAPL')
        self.Schedule.On(self.DateRules.EveryDay('AAPL'), self.TimeRules.AfterMarketOpen('AAPL', 5), self.Predict)
        self.Schedule.On(self.DateRules.EveryDay('AAPL'), self.TimeRules.AfterMarketOpen('AAPL', 6), self.PlotMe)
        
        # Create custom charts
        prediction = Chart('Prediction Plot')
        prediction.AddSeries(Series('Actual Bull', SeriesType.Line, 0))
        prediction.AddSeries(Series('Predicted Bull', SeriesType.Line, 0))
        
        prediction.AddSeries(Series('Actual Bear', SeriesType.Line, 1))
        prediction.AddSeries(Series('Predicted Bear', SeriesType.Line, 1))
        
        
    
        
    def TrainMyModel(self):
    
        qb = self
        #full history of close price
        
        for key, symbol in self.macro_symbols.items():
       
            #full history of close price
            full = qb.History([symbol], 1479, Resolution.Daily)
            #close price 
            close = full.loc[symbol].close[-1280: ]
            #SMA 200
            sma200 = full.loc[symbol].close[:1479 ]
            sma200 = sma200.rolling(window=200).mean()
            sma200.dropna()
            #SMA50
            sma50 = full.loc[symbol].close[-1330: ]
            sma50 = sma50.rolling(window=50).mean()
            sma50.dropna()
            
            df = {'50': sma50,
                    '200': sma200,'close':close }
            df = pd.DataFrame(df)
            df = df.dropna()
                 
                 
                    
        # Iterate over macro symbols
        for key, symbol in self.macro_symbols.items():
            # Initialize LSTM class instance
            lstm = MyLSTM()
            # Prepare data
            features_set, labels, training_data, test_data = lstm.ProcessData(df)
            # Build model layers
            lstm.CreateModel(features_set, labels)
            # Fit model
            lstm.FitModel(features_set, labels)
            # Add LSTM class to dictionary to store later
            self.models[key] = lstm
            
            
            

    def Predict(self):
        
        qb = self
        delta = {}
        
        for key, symbol in self.macro_symbols.items():
            # Fetch LSTM class
            lstm = self.models[key]
            # Fetch history

            #full history of close price
            full = qb.History([symbol], 1479, Resolution.Daily)
            #close price 
            close = full.loc[symbol].close[-1280: ]
            #SMA 200
            sma200 = full.loc[symbol].close[:1479 ]
            sma200 = sma200.rolling(window=200).mean()
            sma200.dropna()
            #SMA50
            sma50 = full.loc[symbol].close[-1330: ]
            sma50 = sma50.rolling(window=50).mean()
            sma50.dropna()
            
            df = {'50': sma50,
                    '200': sma200,'close':close }
            df = pd.DataFrame(df)
            df = df.dropna()
                
             
            test_close = df['close']
                
            # Predict
            predictions = lstm.PredictFromModel(df,test_close)
            
            
            # Grab latest prediction and calculate if predict symbol to go up or down
            delta[key] = ( predictions[-1] / self.Securities[symbol].Price ) - 1
            
            
            # Plot prediction
            self.Plot('Prediction Plot', f'Predicted {key}', predictions[-1])
            
            
        insights = []
        # Iterate over macro symbols
        for key, change in delta.items():
            if key == 'Bull':
                insights += [Insight.Price(symbol, timedelta(1), InsightDirection.Up if change > 0 else InsightDirection.Flat) for symbol in LiquidETFUniverse.SP500Sectors.Long if self.Securities.ContainsKey(symbol)]
                insights += [Insight.Price(symbol, timedelta(1), InsightDirection.Up if change > 0 else InsightDirection.Flat) for symbol in LiquidETFUniverse.Treasuries.Inverse if self.Securities.ContainsKey(symbol)]
                insights += [Insight.Price(symbol, timedelta(1), InsightDirection.Flat if change > 0 else InsightDirection.Up) for symbol in LiquidETFUniverse.Treasuries.Long if self.Securities.ContainsKey(symbol)]
        self.EmitInsights(insights)
        
    def PlotMe(self):
        # Plot current price of symbols to match against prediction
        for key, symbol in self.macro_symbols.items():
            self.Plot('Prediction Plot', f'Actual {key}', self.Securities[symbol].Price)
