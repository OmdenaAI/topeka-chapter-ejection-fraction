import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_results(y_true, y_pred):
    plt.style.use('ggplot')

    plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted')
    plt.hist(y_true, bins=20, alpha=0.5, label='Actual')
    plt.legend(loc='upper right')
    plt.xlabel('EF')
    plt.ylabel('Count')
    plt.title('Histogram of Predicted vs Actual Values')

    mu, std = stats.norm.fit(y_pred)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.show()
