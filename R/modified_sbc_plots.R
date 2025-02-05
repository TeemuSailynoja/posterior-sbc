# This file defines re-themed versions of the bayesplot::ppc_pit_ecdf_grouped
# and the SBC::plot_sim_estimated functions.

modified_ppc_pit_ecdf_grouped <- function(
    y, yrep, group, ..., K = NULL, pit = NULL, prob = 0.99,
    plot_diff = FALSE, interpolate_adj = NULL, facet_labeller = NULL, nrow = 2) {
  bayesplot:::check_ignored_arguments(..., ok_args = c(
    "K", "pit", "prob",
    "plot_diff", "interpolate_adj"
  ))
  require("dplyr")
  if (is.null(pit)) {
    pit <- bayesplot::ppc_data(y, yrep, group) %>%
      group_by(.data$y_id) %>%
      group_map(~ mean(.x$value[.x$is_y] > .x$value[!.x$is_y]) +
        runif(1, max = mean(.x$value[.x$is_y] == .x$value[!.x$is_y]))) %>%
      unlist()
    if (is.null(K)) {
      K <- min(nrow(yrep) + 1, 1000)
    }
  } else {
    rlang::inform("'pit' specified so ignoring 'y' and 'yrep' if specified.")
    pit <- bayesplot:::validate_pit(pit)
  }
  N <- length(pit)
  gammas <- lapply(unique(group), function(g) {
    N_g <- sum(group == g)
    bayesplot:::adjust_gamma(
      N = N_g, K = ifelse(is.null(K), N_g, K),
      prob = prob, interpolate_adj = interpolate_adj
    )
  })
  names(gammas) <- unique(group)
  data <- data.frame(pit = pit, group = group) %>%
    group_by(group) %>%
    group_map(~ data.frame(
      ecdf_value = ecdf(.x$pit)(seq(0,
        1,
        length.out = ifelse(is.null(K), nrow(.x), K)
      )),
      group = .y[1], lims_upper = bayesplot:::ecdf_intervals(
        gamma = gammas[[unlist(.y[1])]],
        N = nrow(.x), K = ifelse(is.null(K), nrow(.x),
          K
        )
      )$upper[-1] / nrow(.x), lims_lower = bayesplot:::ecdf_intervals(
        gamma = gammas[[unlist(.y[1])]],
        N = nrow(.x), K = ifelse(is.null(K), nrow(.x),
          K
        )
      )$lower[-1] / nrow(.x), x = seq(0, 1, length.out = ifelse(is.null(K),
        nrow(.x), K
      ))
    )) %>%
    bind_rows()
  ggplot(data) +
    aes(x = .data$x, y = .data$ecdf_value - (plot_diff ==
      TRUE) * .data$x, group = .data$group, color = "y") +
    geom_step(show.legend = FALSE) +
    geom_step(
      aes(y = .data$lims_upper -
        (plot_diff == TRUE) * .data$x, color = "yrep"),
      linetype = 2,
      show.legend = FALSE
    ) +
    geom_step(
      aes(y = .data$lims_lower -
        (plot_diff == TRUE) * .data$x, color = "yrep"),
      linetype = 2,
      show.legend = FALSE
    ) +
    labs(y = ifelse(plot_diff, "ECDF difference",
      "ECDF"
    ), x = "PIT") +
    bayesplot:::yaxis_ticks(FALSE) +
    bayesplot::bayesplot_theme_get() +
    facet_wrap("group", labeller = facet_labeller, nrow = nrow) +
    bayesplot:::scale_color_ppc() +
    bayesplot:::force_axes_in_facets()
}

modified_plot_sim_estimated <- function(
    x,
    variables = NULL,
    estimate = "mean",
    uncertainty = c("q5", "q95"),
    alpha = NULL,
    parameters = NULL,
    facet_labeller = NULL,
    nrow = 2) {
  require(ggplot2)
  if (!is.null(parameters)) {
    warning("The `parameters` argument is deprecated use `variables` instead.")
    if (is.null(variables)) {
      variables <- parameters
    }
  }
  if ("parameter" %in% names(x)) {
    if (!("variable" %in% names(x))) {
      warning("The x parameter contains a `parameter` column, which is deprecated, use `variable` instead.")
      x$variable <- x$parameter
    }
  }
  required_columns <- c("variable", estimate, uncertainty)
  if (!all(required_columns %in% names(x))) {
    stop(
      "The data.frame needs to have the following columns: ",
      paste0("'", required_columns, "'", collapse = ", ")
    )
  }
  if (!is.null(variables)) {
    x <- dplyr::filter(x, variable %in% variables)
  }
  if (is.null(alpha)) {
    n_points <- dplyr::summarise(dplyr::group_by(x, variable),
      count = dplyr::n()
    )
    max_points <- max(n_points$count)
    alpha_guess <- 1 / ((max_points * 0.06) + 1)
    alpha <- max(0.05, alpha_guess)
  }
  x$estimate__ <- x[[estimate]]
  if (length(uncertainty) != 2) {
    stop("'uncertainty' has to be null or a character vector of length 2")
  }
  x$low__ <- x[[uncertainty[1]]]
  x$high__ <- x[[uncertainty[2]]]
  all_aes <- aes(
    x = simulated_value, y = estimate__, ymin = low__,
    ymax = high__
  )
  y_label <- paste0(
    "posterior ", estimate, " (", uncertainty[1],
    " - ", uncertainty[2], ")"
  )
  if (nrow(x) == 0) {
    stop("No data to plot.")
  }
  ggplot(x, all_aes) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed"
    ) +
    geom_linerange(
      alpha = alpha,
      linewidth = .6,
      color = "#AAAAAA"
    ) +
    geom_point(
      stroke = 0,
      shape = 21,
      fill = "#386cb0",
      color = "#386cb0",
      size = 1.2
    ) +
    labs(y = y_label) +
    facet_wrap(
      ~variable,
      scales = "free",
      labeller = facet_labeller,
      nrow = nrow
    )
}
