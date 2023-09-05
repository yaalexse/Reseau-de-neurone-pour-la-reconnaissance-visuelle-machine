

% Typeset piecewise functions using cases
\documentclass{article}

% Required package
\usepackage{amsmath,amssymb}

\begin{document}

\begin{equation}
\chi_{\mathbb{Q}}(x)=
    \begin{cases}
        1 & \text{if } x \in \mathbb{Q}\\
        0 & \text{if } x \in \mathbb{R}\setminus\mathbb{Q}
    \end{cases}
\end{equation}

\end{document}