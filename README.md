# minesweeper

I was quarantined with some friends because of COVID-19, and we re-discovered the Minesweeper game out of boredom. 
After several games (and many losses), we suggested tohat we should try to implement some algorithms to solve this game.

What began as a joke soon began a real challenge, in which some other friends of mine participated: the challenge was born.
The rules? Try to implement a Minesweeper solver as performant as possible (in terms of success rate, we did not consider complexity/optimization) in less than 72 hours.
To benchmark our results, we compared our results on two grids:
* a 16\*16 grid with 40 mines, known as the "Stanford" grid
* a 16\*32 grid with 99 mines, known as the "Expert" grid


I worked in close collaboration with a good friend of mine, Paul Micoud.

Our algorithm comes in 3 (and a half?!) versions, the latest presenting the best performance (but being very slow, compared to the others). The following presentation sums up the outline the strategies developed in each version: https://docs.google.com/presentation/d/1GE8A1S1u2g5CViuBVn43bpeWqDXg4BTwlHqJPMrrF-g/edit?usp=sharing


The challenge being timed, we later came up with several improvements:
* The 3rd version sometimes raises an error - it needs some troubleshooting
* The algorithm obviously needs some optimization/refactoring
* It may be interesting, when blocked, to randomly click on hidden non-edges when the number of mines is so small that their proba to be a mine becomes smaller than those of hidden edges
* If we could have some kind of measure of the potential information given by a certain hidden cell, we could choose which cell to click on rather than choosing purely randomly. This would be a really complex, but assuredly useful implementation!

We may implement those improvements in the near future.
