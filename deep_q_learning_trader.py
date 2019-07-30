import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 6
        self.action_size = 4
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        self.called_once = False
        self.gamma = 0.1
        self.memory_portfolio = deque(maxlen=2)
        self.memory_state = deque(maxlen=2)
        self.memory_action = None

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action_a = None
        self.last_action_b = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        # TODO Compute the current state
        def get_state():
            expert_vot_dict = {"BUY": 1, "SELL": 2, "HOLD": 0}
            stock_data_a = stock_market_data[Company.A]
            stock_data_b = stock_market_data[Company.B]
            vote_a = self.expert_a.vote(stock_data_a)
            vote_a = str(vote_a).split(".")[1]
            vote_b = self.expert_b.vote(stock_data_b)
            vote_b = str(vote_b).split(".")[1]
            vote_a = expert_vot_dict[vote_a]
            vote_b = expert_vot_dict[vote_b]
            state = np.array([portfolio.get_stock(Company.A), portfolio.get_stock(Company.B), portfolio.cash,
                              portfolio.get_value(stock_market_data), vote_a, vote_b])
            state = state.reshape((1, self.state_size))
            return state

        # TODO Store state as experience (memory) and train the neural network only if trade() was called before at least once
        def train_without_experience_replay(current_state, next_state, reward, action_index):
            expected_value = reward + self.gamma * np.amax(self.model.predict(next_state))
            expected_value_array = self.model.predict(current_state)
            expected_value_array[0][action_index] = expected_value
            self.model.fit(current_state, expected_value_array, epochs=1, verbose=0)

        # TODO Save created state, actions and portfolio value for the next call of trade()
        def experience_replay():
            batch = random.sample(self.memory, self.batch_size)
            for state, action_index, reward, next_state in batch:
                expected_value_array = self.model.predict(state)
                expected_value = reward + self.gamma * np.amax(self.model.predict(next_state))
                expected_value_array[0][action_index] = expected_value
                self.model.fit(state, expected_value_array, epochs=1, verbose=0)

        # TODO Create actions for current state and decrease epsilon for fewer random actions
        def get_order_index(state):
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            if np.random.random() < self.epsilon:
                return random.randrange(self.action_size)
            return np.argmax(self.model.predict(state)[0])

        def get_orders(order_index):
            order_combination_dict = {0: ["BUY", "BUY"], 1: ["SELL", "SELL"], 2: ["BUY", "SELL"], 3: ["SELL", "BUY"]}
            order_dict = {"BUY": 1, "SELL": 2}
            action_combination = order_combination_dict[order_index]
            order_a = order_dict[action_combination[0]]
            order_b = order_dict[action_combination[1]]
            order_a = Order(OrderType(order_a), Company.A, portfolio.get_stock(Company.A))
            order_b = Order(OrderType(order_b), Company.B, portfolio.get_stock(Company.B))
            order_list = [order_a, order_b]
            return order_list

        def follow_orders(orders):
            order_list = []
            company_list = stock_market_data.get_companies()
            for company, order in zip(company_list, orders):
                stock_data = stock_market_data[company]
                if order.type == OrderType.BUY:
                    # buy as many stocks as possible
                    stock_price = stock_data.get_last()[-1]
                    amount_to_buy = int(portfolio.cash // stock_price)
                    logger.debug(f"{self.get_name()}: Got order to buy {company}: {amount_to_buy} shares a {stock_price}")
                    if amount_to_buy > 0:
                        order_list.append(Order(OrderType.BUY, company, amount_to_buy))
                elif order.type == OrderType.SELL:
                    # sell as many stocks as possible
                    amount_to_sell = portfolio.get_stock(company)
                    logger.debug(f"{self.get_name()}: Got order to sell {company}: {amount_to_sell} shares available")
                    if amount_to_sell > 0:
                        order_list.append(Order(OrderType.SELL, Company.A, amount_to_sell))
            return order_list

        def get_reward(current_portfolio_value, next_portfolio_value):
            if next_portfolio_value > current_portfolio_value:
                reward = 1
                return reward
            elif next_portfolio_value < current_portfolio_value:
                reward = -2
                return reward
            else:
                reward = -1
                return reward

        self.last_state = get_state()
        self.memory_state.append(self.last_state)
        self.last_portfolio_value = portfolio.get_value(stock_market_data)
        self.memory_portfolio.append(self.last_portfolio_value)
        if self.called_once:
            current_state = self.memory_state[0]
            next_state = self.memory_state[1]
            reward = get_reward(self.memory_portfolio[0], self.memory_portfolio[1])
            self.memory.append((current_state, self.memory_action, reward, next_state))
            if len(self.memory) >= self.min_size_of_memory_before_training:
                experience_replay()
            self.last_state = next_state
        action_index = get_order_index(self.last_state)
        action_list = get_orders(action_index)
        action_list = follow_orders(action_list)
        self.memory_action = action_index
        self.called_once = True
        return action_list


# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":

    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()


