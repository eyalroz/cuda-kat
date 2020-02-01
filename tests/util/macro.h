#ifndef TESTS_UTIL_MACRO_H_
#define TESTS_UTIL_MACRO_H_


#if defined(__GNUC__) && __GNUC__ >= 4
#ifndef UNLIKELY
#define LIKELY(x)   (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif /* UNLIKELY */
#else /* defined(__GNUC__) && __GNUC__ >= 4 */
#ifndef UNLIKELY
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif /* UNLIKELY */
#endif /* defined(__GNUC__) && __GNUC__ >= 4 */

#ifndef UNUSED
#define UNUSED(x) (void) x
#endif

#define EXPAND(_x)                          _x
#define QUOTE(_q)                           #_q
#define STRINGIZE(_q)                       #_q

#ifndef CONCATENATE
#define CONCATENATE( s1, s2 )               s1 ## s2
#define EXPAND_THEN_CONCATENATE( s1, s2 )   CONCATENATE( s1, s2 )
#endif /* CONCATENATE */

#define AS_SINGLE_ARGUMENT(...)             __VA_ARGS__

/**
 * This macro expands into a different identifier in every expansion.
 * Note that you _can_ clash with an invocation of UNIQUE_IDENTIFIER
 * by manually using the same identifier elsewhere; or by carefully
 * choosing another prefix etc.
 */
#ifdef __COUNTER__
#define UNIQUE_IDENTIFIER(prefix) EXPAND_THEN_CONCATENATE(prefix, __COUNTER__)
#else
#define UNIQUE_IDENTIFIER(prefix) EXPAND_THEN_CONCATENATE(prefix, __LINE__)
#endif /* COUNTER */

#define COUNT_THIS_LINE static_assert(__COUNTER__ + 1, "");
#define START_COUNTING_LINES(count_name) enum { EXPAND_THEN_CONCATENATE(count_name,_start) = __COUNTER__ };
#define FINISH_COUNTING_LINES(count_name) enum { count_name = __COUNTER__ - EXPAND_THEN_CONCATENATE(count_name,_start) - 1 };


///**
// * This macro expands into a different identifier in every expansion.
// * Note that you _can_ clash with an invocation of UNIQUE_IDENTIFIER
// * by manually using the same identifier elsewhere; or by carefully
// * choosing another prefix etc.
// */
//#ifdef __COUNTER__
//#define UNIQUE_IDENTIFIER(prefix) EXPAND_THEN_CONCATENATE(prefix, __COUNTER__)
//#else
//#define UNIQUE_IDENTIFIER(prefix) EXPAND_THEN_CONCATENATE(prefix, __LINE__)
//#endif /* COUNTER */


/**
 * Map macro - applying an arbitrary macro to multiple arguments;
 * based on the discussion and William Swanson's suggestion here:
 * http://stackoverflow.com/q/6707148/1593077
 *
 * Usage example:
 *
 *   #define DO_SOMETHING(x) char const *x##_string = #x;
 *   MAP(DO_SOMETHING, foo, bar, baz)
 *
 * will expand to
 *
 *    char const *foo_string = "foo";
 *    char const *bar_string = "bar";
 *    char const *baz_string = "baz";
 *
 */

#define EVAL0(...) __VA_ARGS__
#define EVAL1(...) EVAL0 (EVAL0 (EVAL0 (__VA_ARGS__)))
#define EVAL2(...) EVAL1 (EVAL1 (EVAL1 (__VA_ARGS__)))
#define EVAL3(...) EVAL2 (EVAL2 (EVAL2 (__VA_ARGS__)))
#define EVAL4(...) EVAL3 (EVAL3 (EVAL3 (__VA_ARGS__)))
#define EVAL(...)  EVAL4 (EVAL4 (EVAL4 (__VA_ARGS__)))

#define MAP_END(...)
#define MAP_OUT

#define MAP_GET_END() 0, MAP_END
#define MAP_NEXT0(test, next, ...) next MAP_OUT
#define MAP_NEXT1(test, next) MAP_NEXT0 (test, next, 0)
#define MAP_NEXT(test, next)  MAP_NEXT1 (MAP_GET_END test, next)

/**
 *  Use the third of these macros to apply a unary macro to all other arguments
 * passed, e.g.
 *
 *   #define MY_UNARY(x)   call_foo(x, 123)
 *   MAP(MY_UNARY, 456, 789);
 *
 * will expand to
 *
 *   call_foo(456, 123);
 *   call_foo(789, 123);
 *
 */
#define MAP0(f, x, peek, ...) f(x) MAP_NEXT (peek, MAP1) (f, peek, __VA_ARGS__)
#define MAP1(f, x, peek, ...) f(x) MAP_NEXT (peek, MAP0) (f, peek, __VA_ARGS__)
#define MAP(f, ...) EVAL (MAP1 (f, __VA_ARGS__, (), 0))

/**
 * Same as MAP/MAP1/MAP0, but used for macros with pairs of arguments, and
 * specifying the first one
 */
#define MAP_BINARY0(f, fixed_arg, x, peek, ...) f(fixed_arg, x) MAP_NEXT (peek, MAP_BINARY1) (f, fixed_arg, peek, __VA_ARGS__)
#define MAP_BINARY1(f, fixed_arg, x, peek, ...) f(fixed_arg, x) MAP_NEXT (peek, MAP_BINARY0) (f, fixed_arg, peek, __VA_ARGS__)
#define MAP_BINARY(f, fixed_arg, ...) EVAL (MAP_BINARY1 (f, fixed_arg, __VA_ARGS__, (), 0))

/**
 * Same as MAP/MAP1/MAP0, but used for macros with triplets of arguments, and
 * specifying the first and second ones
 */
#define MAP_TRINARY0(f, first_fixed_arg, second_fixed_arg, x, peek, ...) f(first_fixed_arg, second_fixed_arg, x) MAP_NEXT (peek, MAP_TRINARY1) (f, first_fixed_arg, second_fixed_arg, peek, __VA_ARGS__)
#define MAP_TRINARY1(f, first_fixed_arg, second_fixed_arg, x, peek, ...) f(first_fixed_arg, second_fixed_arg, x) MAP_NEXT (peek, MAP_TRINARY0) (f, first_fixed_arg, second_fixed_arg, peek, __VA_ARGS__)
#define MAP_TRINARY(f, first_fixed_arg, second_fixed_arg, ...) EVAL (MAP_TRINARY1 (f, first_fixed_arg, second_fixed_arg, __VA_ARGS__, (), 0))

/**
 * Compile a different piece of code based on compile-time evaluation of a condition;
 * the condition must evaluate to 1 or to 0, exactly, or this will fail.
 *
 * Usage:
 *
 *   IF_ELSE( GCC_VERSION > 4 )(code in case condition holds)(code in case condition fails)
 */

#define IF_ELSE(condition) _IF_ ## condition
#define _IF_1(...) __VA_ARGS__ _IF_1_ELSE
#define _IF_0(...)             _IF_0_ELSE

#define _IF_1_ELSE(...)
#define _IF_0_ELSE(...) __VA_ARGS__

#endif // TESTS_UTIL_MACRO_H_
