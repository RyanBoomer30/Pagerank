import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    result = dict()

    for p in corpus:
        # Set up initual randomizer value
        initual = (1-damping_factor) / len(corpus)

        # Add initualize value to chances of connected node
        for i in corpus[page]:
            if p == i:
                initual += (damping_factor / len(corpus[page]))

        # Add initualize value to a randomizer value if no connected node
        if len(corpus[page]) == 0:
            initual += damping_factor / len(corpus)
        
        result[p] = initual
    
    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initualize all page as 0
    result = dict()
    for i in corpus:
        result[i] = 0

    # Choose a random first page
    currentPage_name = list(corpus.keys())
    currentPage = random.choice(currentPage_name)

    for i in range(n):

        result[currentPage] += 1

        # Get output from the current move
        moves = transition_model(corpus, currentPage, damping_factor)

        # Choose a random move based on probability of moves
        moves_name = list(moves.keys())
        moves_chances = list(moves.values())
        currentPage = random.choices(moves_name, weights=moves_chances, k=1)
        currentPage = currentPage[0]

    for i in corpus:
        result[i] = result[i] / n

    return result
        

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initualize all page as 1 / N
    previous_Iteration = dict()
    for i in corpus:
        previous_Iteration[i] = 1 / len(corpus)

    while True:
        check = 0
        
        for i in corpus:
            current_Iteration = (1-damping_factor) / len(corpus)
            sum_Iteration = 0

            # Sum up all the nodes that link to the current page
            for p in corpus:
                if i in corpus[p]:
                    sum_Iteration += previous_Iteration[p] / len(corpus[p])

            current_Iteration += damping_factor * sum_Iteration

            # Check if a Pagerank value changes by more than 0.001
            if abs(previous_Iteration[i] - current_Iteration <= 0.001):
                check +=1

            previous_Iteration[i] = current_Iteration
        
        # Stop the loop if all Pagerank values changes by more than 0.001
        if check == len(corpus):
            # Normalize the value to get a perfect sum of 1 (Chris Sorenson's idea)
            normalizer = sum(previous_Iteration.values())
            for i in previous_Iteration:
                previous_Iteration[i] /= normalizer
            print(sum(previous_Iteration.values()))
            return previous_Iteration


if __name__ == "__main__":
    main()
