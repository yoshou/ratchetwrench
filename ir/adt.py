from enum import Enum, Flag, auto

CHAR_BIT = 8
APINT_WORD_SIZE = 8
APINT_BITS_PER_WORD = APINT_WORD_SIZE * CHAR_BIT

class APInt:
    def clone(self):
        cloned = APInt(self.num_bits, self.is_signed)
        cloned.values = list(self.values)
        return cloned
    
    def __init__(self, num_bits, is_signed=False):
        self.num_bits = num_bits
        self.values = [0] * self.num_words
        self.is_signed = is_signed

    def get_bit(self, position):
        if position >= self.num_bits:
            raise IndexError("The position is out of range.")

        word_idx = int(position / APINT_BITS_PER_WORD)
        bit_idx = position % APINT_BITS_PER_WORD
        return (self.values[word_idx] >> bit_idx) & 0x1

    def set_bit(self, position, value):
        if position >= self.num_bits:
            raise IndexError("The position is out of range.")

        word_idx = int(position / APINT_BITS_PER_WORD)
        bit_idx = position % APINT_BITS_PER_WORD

        value = 0x1 if value != 0 else 0x0
        mask = (value << bit_idx)
        self.values[word_idx] = (self.values[word_idx] & ~mask) | mask

    @property
    def sign(self):
        return self.is_signed & (self.get_bit(self.num_bits - 1) != 0)

    def to_string(self, radix=10):
        if self.num_words == 1:
            if self.is_signed:
                return str(self.values[0])
            else:
                return str(self.values[0] & ((1 << APINT_BITS_PER_WORD) - 1))

        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".encode()
        s = bytearray()

        if self.is_signed:
            raise NotImplementedError()

        tmp = self
        while not tmp.is_zero:
            digit = (tmp % radix).values[0]
            tmp = tmp / radix
            assert(digit >= 0 and digit <= 9)
            s.append(digits[digit])

        return bytes(reversed(s)).decode()

    def __str__(self):
        return self.to_string()

    @property
    def lsb(self):
        def get_msl_in_word(value, num_bits):
            for i in range(num_bits):
                if (value >> i) & 0x1 != 0:
                    return i

            return -1


        for i in range(self.num_words):
            if self.values[i] != 0:
                msl_in_word = get_msl_in_word(self.values[i], APINT_BITS_PER_WORD)
                return i * APINT_BITS_PER_WORD + msl_in_word

        return -1

    @property
    def msb(self):
        def get_msb_in_word(value, num_bits):
            for i in reversed(range(num_bits)):
                if (value >> i) & 0x1 != 0:
                    return i

            return -1


        for i in reversed(range(self.num_words)):
            if self.values[i] != 0:
                msb_in_word = get_msb_in_word(self.values[i], APINT_BITS_PER_WORD)
                return i * APINT_BITS_PER_WORD + msb_in_word

        return -1

    def set_from_int(self, i: int):
        words = []
        while i != 0:
            words.append(i & self.word_mask)
            i = i >> self.word_shift
            
        if len(words) > len(self.values):
            raise OverflowError()

        self.values[:len(words)] = words

    @property
    def num_words(self):
        return int((self.num_bits + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD)

    @property
    def word_mask(self):
        return ((0x1 << APINT_BITS_PER_WORD) - 1)

    @property
    def word_shift(self):
        return APINT_BITS_PER_WORD

    def _set_word(self, index, value):
        if index >= self.num_words:
            raise IndexError()

        if value >= (1 << APINT_BITS_PER_WORD):
            raise OverflowError()
        
        self.values[index] = value

    def _get_word(self, index):
        if index >= self.num_words:
            raise IndexError()
        
        return self.values[index]

    def compare(self, other):
        assert(self.num_words == other.num_words)
        lhs_vals = self.values
        rhs_vals = other.values

        for i in reversed(range(self.num_words)):
            if lhs_vals[i] != rhs_vals[i]:
                return 1 if lhs_vals[i] > rhs_vals[i] else -1

        return 0

    def __gt__(self, other):
        cmp = self.compare(other)
        return cmp == 1
    
    def __ge__(self, other):
        cmp = self.compare(other)
        return cmp == 1 or cmp == 0

    def __lt__(self, other):
        return not self.__ge__(other)

    def __le__(self, other):
        return not self.__gt__(other)

    @property
    def is_zero(self):
        for val in self.values:
            if val != 0:
                return False

        return True

    def udivrem(self, dividend, divisor, div, rem):
        if divisor.is_zero:
            raise ZeroDivisionError()

        if dividend < divisor:
            div.set_from_int(0)
            rem.values[:] = dividend.values[:]
            return

        if dividend == divisor:
            div.set_from_int(1)
            rem.set_from_int(0)
            return


        shift_cnt = dividend.msb - divisor.msb
        shift_rhs = divisor << shift_cnt

        for i in reversed(range(0, shift_cnt + 1)):
            if dividend >= shift_rhs:
                dividend = dividend - shift_rhs
                div.set_bit(i, 1)

            shift_rhs = shift_rhs >> 1

        rem.values[:] = dividend.values[:]

    def multiply(self, a, b, c):
        num_bits = c.num_bits

        for i in range(b.num_words):
            mult = b.values[i]
            carry = 0
            for j in range(a.num_words):
                a_val = a.values[j]
                c_val = c._get_word(i + j) if (i + j) < c.num_words else 0

                val = a_val * mult + c_val + carry

                # (n - 1) * (n - 1) + 2(n - 1) = (n + 1) * (n - 1) = n^2 - 1

                assert(val < (1 << (2 * a.word_shift)))

                lo = val & a.word_mask
                hi = (val >> a.word_shift) & a.word_mask

                if lo != 0 and (i + j >= c.num_words):
                    raise OverflowError()

                if i + j < c.num_words:
                    c._set_word(i + j, lo)
                    
                carry = hi

            if a.num_words + i < c.num_words:
                c._set_word(a.num_words + i, carry)
        
    def __mul__(self, other):
        if isinstance(other, int):
            apint = APInt(self.num_bits, self.is_signed)
            apint.set_from_int(other)
            other = apint

        result = APInt(self.num_bits, self.is_signed)
        self.multiply(self, other, result)
        return result
        
    def __truediv__(self, other):
        if isinstance(other, int):
            apint = APInt(self.num_bits, self.is_signed)
            apint.set_from_int(other)
            other = apint

        div_result = APInt(self.num_bits, self.is_signed)
        rem_result = APInt(self.num_bits, self.is_signed)
        self.udivrem(self, other, div_result, rem_result)
        return div_result
        
    def __mod__(self, other):
        if isinstance(other, int):
            apint = APInt(self.num_bits, self.is_signed)
            apint.set_from_int(other)
            other = apint

        div_result = APInt(self.num_bits, self.is_signed)
        rem_result = APInt(self.num_bits, self.is_signed)
        self.udivrem(self, other, div_result, rem_result)
        return rem_result

    def add(self, a, b, c, carry=0):
        for i in range(a.num_words):
            a_val = a.values[i]
            b_val = b.values[i]

            val = a_val + b_val + carry

            c._set_word(i, val & a.word_mask)

            carry = 1 if val & a.word_mask < a_val else 0
                
        if carry != 0:
            raise OverflowError()

    def __add__(self, other):
        if isinstance(other, int):
            apint = APInt(self.num_bits, self.is_signed)
            apint.set_from_int(other)
            other = apint

        result = APInt(self.num_bits, self.is_signed)
        self.add(self, other, result)
        return result

    def sub(self, a, b, c, carry=0):
        for i in range(a.num_words):
            a_val = a.values[i]
            b_val = b.values[i]

            val = a_val - b_val - carry

            c._set_word(i, val & a.word_mask)

            carry = 1 if val & a.word_mask > a_val else 0
                
        if carry != 0:
            raise OverflowError()

    def __sub__(self, other):
        if isinstance(other, int):
            apint = APInt(self.num_bits, self.is_signed)
            apint.set_from_int(other)
            other = apint

        result = APInt(self.num_bits, self.is_signed)
        self.sub(self, other, result)
        return result

    def __rshift__(self, amount):
        return self.shift_right(amount)

    def __lshift__(self, amount):
        return self.shift_left(amount)

    def shift_right(self, amount):
        result = APInt(self.num_bits, self.is_signed)

        word_amount = int(amount / APINT_BITS_PER_WORD)
        bit_amount = int(amount % APINT_BITS_PER_WORD)

        word_to_move = self.num_words - word_amount
        if bit_amount == 0:
            for i in range(word_to_move):
                result.values[i] = self.values[i + word_amount]
            return result

        for i in range(word_to_move):
            result.values[i] = self.values[i + word_amount] >> bit_amount
            if i + 1 != word_to_move:
                result.values[i] |= (self.values[i + 1 + word_amount] & ((0x1 << bit_amount) - 1)) << (APINT_BITS_PER_WORD - bit_amount)
        return result

    def shift_left(self, amount):
        result = APInt(self.num_bits, self.is_signed)

        word_amount = int(amount / APINT_BITS_PER_WORD)
        bit_amount = int(amount % APINT_BITS_PER_WORD)

        if bit_amount == 0:
            for i in range(self.num_words - word_amount):
                result.values[i + word_amount] = self.values[i]
            return result

        for i in reversed(range(word_amount, self.num_words)):
            result.values[i] = (self.values[i - word_amount] << bit_amount) & ((0x1 << APINT_BITS_PER_WORD) - 1)
            if i > word_amount:
                result.values[i] |= self.values[i - word_amount - 1] >> (APINT_BITS_PER_WORD - bit_amount)
        return result

    def set_num_bits(self, num_bits):
        new_value = APInt(num_bits, self.is_signed)
        copy_words = min(self.num_words, new_value.num_words)
        new_value.values[:copy_words] = self.values[:copy_words]
        self.values = new_value.values
        self.num_bits = new_value.num_bits

        if self.is_signed:
            raise NotImplementedError() # Need sign extension

    def set_from_string(self, s: str, radix: int):
        data = s.encode('ascii')

        pos = 0
        is_neg = data[pos] == ord('-')
        if data[pos] in [ord('-'), ord('+')]:
            pos += 1
            assert(pos < len(data))
            
        while pos < len(data):
            val = 0
            multiplier = 1
            max_multiplier = int(((1 << self.word_shift) - 1 + radix - 1) / radix)
            while multiplier < max_multiplier and pos < len(data):
                digit = int(data[pos]) - ord('0')
                assert(digit <= 9 and digit >= 0)
                pos += 1
                multiplier *= radix
                val = val * radix + digit

            self.values = (self * multiplier + val).values

    def extract_to(self, to, src_bits, src_lsb):
        assert(isinstance(to, APInt))
        #assert(self.num_words <= to.num_words)

        # src_l = src_lsb / APINT_BITS_PER_WORD
        # src_h = src_l + int((src_bits + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD)

        # to.values[:(src_h - src_l)] = self.values[src_l:src_h]
        shifted = self.shift_right(src_lsb)

        to.values[:self.num_words] = self.shift_right(src_lsb).values[:self.num_words]

    def to_bytes(self, length, byteorder, signed=False):
        bigval = 0
        for val in reversed(self.values):
            bigval = (bigval << APINT_BITS_PER_WORD) | (val & ((1 << APINT_BITS_PER_WORD) - 1))
            
        return bigval.to_bytes(length, byteorder, signed=signed)



class FltSemantics:
    def __init__(self, max_exp, min_exp, precision, num_bits):
        self.max_exp = max_exp
        self.min_exp = min_exp
        self.precision = precision
        self.num_bits = num_bits

class FltSemanticsDef(Enum):
    IEEE_Half = FltSemantics(15, -14, 11, 16)
    IEEE_Single = FltSemantics(127, -126, 24, 32)
    IEEE_Double = FltSemantics(1023, -1022, 53, 64)
    IEEE_Quad = FltSemantics(16383, -16382, 113, 128)

class FltRoundingMode(Enum):
    NearestTiesToEven = auto()
    TowardPositive = auto()
    TowardNegative = auto()
    TowardZero = auto()
    NearestTiesToAway = auto()

class FltExceptions(Flag):
    Ok = 0x00
    InvalidOp = 0x01
    DivByZero = 0x02
    Overflow = 0x04
    Underflow = 0x08
    Inexact = 0x10

class FltCategory(Enum):
    Infinity = auto()
    Nan = auto()
    Normal = auto()
    Zero = auto()

def get_lost_fraction_through_trunc(frac, bits):
    lsb = frac.lsb

    if lsb >= bits:
        return FltLostFraction.ExactlyZero
    elif lsb == bits - 1:
        return FltLostFraction.ExactlyHalf
    elif frac.get_bit(bits - 1) != 0:
        return FltLostFraction.MoreThanHalf
    
    return FltLostFraction.LessThanHalf

# If current loss is zero or half and previous less is not zero, need to add loss slightly.
def combine_lost_fraction(less, more):
    if less != FltLostFraction.ExactlyZero:
        if more == FltLostFraction.ExactlyZero:
            return FltLostFraction.LessThanHalf
        elif more == FltLostFraction.ExactlyHalf:
            return FltLostFraction.MoreThanHalf

    return more



class IEEEFloat:
    def __init__(self, sema: FltSemantics):
        self.sema = sema
        self.sign = 0
        self.significand = APInt(sema.precision + 1, False)
        self.exponent = 0
        self.category = FltCategory.Normal

    
    def round_away_from_zero(self, rounding_mode, lost_frac, bit):
        if rounding_mode == FltRoundingMode.NearestTiesToAway:
            return lost_frac in [FltLostFraction.ExactlyHalf, FltLostFraction.MoreThanHalf]
        elif rounding_mode == FltRoundingMode.NearestTiesToEven:
            if lost_frac == FltLostFraction.MoreThanHalf:
                return True
            
            if lost_frac == FltLostFraction.ExactlyHalf and self.category != FltCategory.Zero:
                return self.significand.get_bit(bit)

            return False
        elif rounding_mode == FltRoundingMode.TowardZero:
            return False
        elif rounding_mode == FltRoundingMode.TowardPositive:
            return self.sign == 0
        elif rounding_mode == FltRoundingMode.TowardNegative:
            return self.sign == 1

        raise ValueError("Invalid rounding mode.")


    def increase_significand(self):
        self.significand = self.significand + 1

    def shift_right_significand(self, bits):
        self.significand = self.significand >> bits
        self.exponent += bits


    def normalize(self, rounding_mode, lost_frac):
        omsb = self.significand.msb + 1

        if omsb > 0:
            exp_change = omsb - self.sema.precision

            if exp_change < 0:
                self.significand = self.significand << -exp_change
                self.exponent += exp_change
                return FltExceptions.Ok

            if exp_change > 0:
                lost_frac2 = get_lost_fraction_through_trunc(self.significand, exp_change)
                self.significand = self.significand >> exp_change
                lost_frac = combine_lost_fraction(lost_frac, lost_frac2)

        if lost_frac == FltLostFraction.ExactlyZero:
            if omsb == 0:
                self.category = FltCategory.Zero
            return FltExceptions.Ok

        if self.round_away_from_zero(rounding_mode, lost_frac, 0):
            self.increase_significand()
            omsb = self.significand.msb + 1

            # If significand overflowed, do renormalization.
            if omsb == self.sema.precision + 1:
                if self.exponent < self.sema.max_exp:
                    self.shift_right_significand(1)
                    return FltExceptions.Inexact

                self.category = FltCategory.Infinity
                return FltExceptions.Overflow | FltExceptions.Inexact

        # Correctly normalized
        if omsb == self.sema.precision:
            return FltExceptions.Inexact

        if omsb == 0:
            self.category = FltCategory.Zero

        return FltExceptions.Underflow | FltExceptions.Inexact



    def convert_from_integer_part(self, integer_part, rounding_mode):
        num_words = self.significand.num_bits
        precision = self.sema.precision
        
        omsb = integer_part.msb + 1
        if precision <= omsb:
            self.exponent = omsb - 1
            lost_frac = get_lost_fraction_through_trunc(integer_part, omsb - precision)
            integer_part.extract_to(self.significand, precision, omsb - precision)
        else:
            self.exponent = precision - 1
            lost_frac = FltLostFraction.ExactlyZero
            integer_part.extract_to(self.significand, precision, 0)

        return self.normalize(rounding_mode, lost_frac)

    def multiply_significand(self, rhs):
        precision = self.sema.precision
        a = self.significand
        b = rhs.significand
        c = APInt(precision * 2 + 1) # Additional overflow bit

        # Need to add 2 to exponent.
        # MSB of (a * b) is (precision - 1) * 2 = precision * 2 - 2, otherwise
        # MSB of c is precision * 2 + 1 - 1 = precision * 2.

        self.exponent += rhs.exponent
        self.exponent += 2

        self.significand.multiply(a, b, c)

        omsb = c.msb + 1
        
        self.exponent -= (precision + 1)

        lost_frac = FltLostFraction.ExactlyZero

        if omsb > precision:
            bits = omsb - precision
            lost_frac2 = get_lost_fraction_through_trunc(c, bits)
            c = c >> bits
            lost_frac = combine_lost_fraction(lost_frac, lost_frac2)
            self.exponent += bits

        self.significand.values[:] = c.values[:self.significand.num_words]

        return lost_frac

    def divide_significand(self, rhs):
        precision = self.sema.precision
        a = self.significand
        b = rhs.significand
        self.exponent -= rhs.exponent

        assert(precision - (a.msb + 1) == 0)
        assert(precision - (b.msb + 1) == 0)

        c = a

        bit = precision - b.msb - 1
        if bit != 0:
            self.exponent += bit
            b = b << bit

        bit = precision - a.msb - 1
        if bit != 0:
            self.exponent -= bit
            a = a << bit

        if b > a:
            a = a << 1
            self.exponent -= 1
            assert(a >= b)

        result = APInt(a.num_bits, a.is_signed)
        for i in reversed(range(precision)):
            if a >= b:
                a = a - b
                result.set_bit(i, 1)

            a = a << 1

        cmp = a.compare(b)
        if cmp > 0:
            lost_frac = FltLostFraction.MoreThanHalf
        elif cmp == 0:
            lost_frac = FltLostFraction.ExactlyHalf
        elif a.is_zero:
            lost_frac = FltLostFraction.ExactlyZero
        else:
            lost_frac = FltLostFraction.LessThanHalf

        c.values[:] = result.values[:]

        return lost_frac


    def _get_pow5(self, power):
        assert(power >= 0)
        pow5 = APInt(self.sema.precision + 11, False)
        pow5.set_from_int(1)
        five = APInt(self.sema.precision + 11, False)
        five.set_from_int(5)
        for i in range(power):
            pow5 = pow5 * five

        return pow5



    def set_significand_with_exp(self, significand, int_part, exp, rounding_mode):
        def get_num_word(num_bits):
            return int((num_bits + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD)

        is_nearest = rounding_mode in [FltRoundingMode.NearestTiesToEven, FltRoundingMode.NearestTiesToAway]

        num_words = get_num_word(self.sema.precision + 11)
        calc_semantics = FltSemantics(32767, -32767, 0, 0)
        pow5_int = self._get_pow5(abs(exp))
        
        while True:
            precision = num_words * APINT_BITS_PER_WORD - 1
            calc_semantics.precision = precision

            decsig = IEEEFloat(calc_semantics)
            pow5 = IEEEFloat(calc_semantics)

            decsig_status = decsig.convert_from_integer_part(significand, FltRoundingMode.NearestTiesToEven)
            pow5_status = pow5.convert_from_integer_part(pow5_int, FltRoundingMode.NearestTiesToEven)

            excess_precision = calc_semantics.precision - self.sema.precision
            truncated_bits = excess_precision

            # 10^n = 2^n * 5^n
            decsig.exponent += exp

            assert(decsig.exponent >= self.sema.min_exp and decsig.exponent <= self.sema.max_exp)

            if exp >= 0:
                lost_frac = decsig.multiply_significand(pow5)
                pow_half_ulp_err = 0 if pow5_status == FltExceptions.Ok else 1
            else:
                lost_frac = decsig.divide_significand(pow5)
                pow_half_ulp_err = 0 if pow5_status == FltExceptions.Ok and lost_frac == FltLostFraction.ExactlyZero else 2

            half_ulp_err = compute_half_ulp_bound(lost_frac != FltLostFraction.ExactlyZero, decsig_status != FltExceptions.Ok, pow_half_ulp_err)
            half_ulp_distance = 2 * compute_ulp_from_boundary(decsig.significand, excess_precision, is_nearest)

            if half_ulp_distance >= half_ulp_err:
                decsig.significand.extract_to(self.significand, calc_semantics.precision - excess_precision, excess_precision)
                self.exponent = (decsig.exponent + self.sema.precision) - (calc_semantics.precision - excess_precision)
            
                lost_frac = get_lost_fraction_through_trunc(decsig.significand, truncated_bits)
                return self.normalize(rounding_mode, lost_frac)
                
            num_words *= 2

    def convert_single_to_apint(self):
        apint = APInt(32)

        sig_value = 0
        for val in reversed(self.significand.values):
            sig_value = (sig_value << APINT_BITS_PER_WORD) | (val & ((0x1 << APINT_BITS_PER_WORD) - 1))

        if self.category == FltCategory.Zero:
            exponent = 0
            significand = 0
        elif self.category == FltCategory.Infinity:
            exponent = 0xff
            significand = 0
        elif self.category == FltCategory.Nan:
            exponent = 0xff
            significand = sig_value
        else:
            exponent = self.exponent + 127
            significand = sig_value

        apint.set_from_int(((self.sign & 0x1) << 31) | ((exponent & 0xff) << 23) | significand & 0x7fffff)
        return apint

    def convert_double_to_apint(self):
        apint = APInt(64)

        sig_value = 0
        for val in reversed(self.significand.values):
            sig_value = (sig_value << APINT_BITS_PER_WORD) | (val & ((0x1 << APINT_BITS_PER_WORD) - 1))

        if self.category == FltCategory.Zero:
            exponent = 0
            significand = 0
        elif self.category == FltCategory.Infinity:
            exponent = 0x7ff
            significand = 0
        elif self.category == FltCategory.Nan:
            exponent = 0x7ff
            significand = sig_value
        else:
            exponent = self.exponent + 1023
            significand = sig_value

        apint.set_from_int(((self.sign & 0x1) << 63) | ((exponent & 0x7ff) << 52) | significand & 0xfffffffffffff)
        return apint

    def bitcast_to_apint(self):
        if self.sema == FltSemanticsDef.IEEE_Half.value:
            return self.convert_double_to_apint()
        elif self.sema == FltSemanticsDef.IEEE_Single.value:
            return self.convert_single_to_apint()
        elif self.sema == FltSemanticsDef.IEEE_Double.value:
            return self.convert_double_to_apint()
        elif self.sema == FltSemanticsDef.IEEE_Quad.value:
            return self.convert_double_to_apint()

        raise NotImplementedError()




class FltLostFraction(Enum):
    ExactlyZero = auto()    # 000000
    LessThanHalf = auto()   # 0xxxxx  x's not all zero
    ExactlyHalf = auto()    # 100000
    MoreThanHalf = auto()    # 1xxxxx  x's not all zero


def compute_half_ulp_bound(inexact_mult, huerr1, huerr2):
    if huerr1 + huerr2 == 0:
        return 2 * inexact_mult
    else:
        return inexact_mult + 2 * (huerr1 + huerr2)

def compute_ulp_from_boundary(integer_part, bits, is_nearest):
    bits -= 1

    count = int(bits / APINT_BITS_PER_WORD)
    part_bits = bits % APINT_BITS_PER_WORD + 1

    parts = integer_part.values
    part = parts[count] & ((0x1 << part_bits) - 1)

    if is_nearest:
        boundary = 1 << (part_bits - 1)
    else:
        boundary = 0

    word_mask = (0x1 << APINT_BITS_PER_WORD) - 1

    if count == 0:
        if (part - boundary) & word_mask <= (boundary - part) & word_mask:
            return part - boundary
        else:
            return boundary - part

    if part == boundary:
        for i in reversed(range(count - 1)):
            if parts ^ ((0x1 << APINT_BITS_PER_WORD) - 1) != 0:
                return ((0x1 << APINT_BITS_PER_WORD) - 1)

        return parts[0]
    elif part == boundary - 1:
        for i in reversed(range(count - 1)):
            if parts ^ ((0x1 << APINT_BITS_PER_WORD) - 1) != 0:
                return ((0x1 << APINT_BITS_PER_WORD) - 1)

        return parts[0] ^ ((0x1 << APINT_BITS_PER_WORD) - 1)
        
    return ((0x1 << APINT_BITS_PER_WORD) - 1)


def is_digit(ch):
    return ch >= ord('0') and ch <= ord('9')


class APFloat:
    def __init__(self, sema: FltSemantics):
        self.value = IEEEFloat(sema)

    def _set_from_hexadecimal_string(self, data: bytes, rounding_mode: FltRoundingMode):
        raise NotImplementedError()

    def get_exponent(self, data: bytes):
        pos = 0
        is_neg = data[pos] == ord('-')
        if data[pos] in [ord('-'), ord('+')]:
            pos += 1
            assert(pos < len(data))

        abs_exp = 0
        while pos < len(data):
            digit = int(data[pos]) - ord('0')
            assert(digit <= 9 and digit >= 0)
            
            abs_exp *= 10
            abs_exp += digit

            pos += 1

        return -abs_exp if is_neg else abs_exp

    def _skip_leading_zeros_and_dot(self, data: bytes):
        pos = 0
        dot = -1
        while pos < len(data) and data[pos] == ord('0'):
            pos += 1

        if pos < len(data) and data[pos] == ord('.'):
            dot = pos
            pos += 1

            while pos < len(data) and data[pos] == ord('0'):
                pos += 1

        return pos, dot

    def _set_from_decimal_string(self, data: bytes, rounding_mode: FltRoundingMode):
        import math

        pos, dot = self._skip_leading_zeros_and_dot(data)

        # Read significand part
        first_digit_pos = pos
        while pos < len(data):
            if data[pos] == ord('.'):
                dot = pos
                pos += 1
                continue

            digit = int(data[pos]) - ord('0')
            if digit < 0 or digit > 9:
                break

            pos += 1

        has_dot = dot > 0 and dot < pos

        # Read exponent part
        exp = 0
        if pos != len(data):
            assert(data[pos] == ord('e') or data[pos] == ord('E'))
            assert(pos + 1 < len(data))

            exp = self.get_exponent(data[pos + 1:])

        # Drop trailing zeros
        while pos - 1 > first_digit_pos and data[pos - 1] == ord('0'):
            pos -= 1
        last_digit_pos = pos

        if has_dot:
            exp = exp - (last_digit_pos - dot - 1)

        if first_digit_pos == len(data) or not is_digit(data[first_digit_pos]):
            self.value.category = FltCategory.Zero
            return FltExceptions.Ok

        pos = first_digit_pos
        dec_digits = last_digit_pos - first_digit_pos
        prec = int(1 + dec_digits * math.log(10) / math.log(2))
        significand = APInt(prec, False)
        while pos < last_digit_pos:
            val = 0
            multiplier = 1
            max_multiplier = int(((1 << significand.word_shift) - 1 + 9) / 10)
            while multiplier < max_multiplier and pos < last_digit_pos:
                if data[pos] == ord('.'):
                    pos += 1

                digit = int(data[pos]) - ord('0')
                assert(digit <= 9 and digit >= 0)
                pos += 1
                multiplier *= 10
                val = val * 10 + digit

            significand = significand * multiplier + val

        return self.value.set_significand_with_exp(significand, dot, exp, rounding_mode)

    @property
    def sign(self):
        return self.value.sign

    @sign.setter
    def sign(self, value):
        self.value.sign = 1 if value != 0 else 0

    @property
    def sema(self):
        return self.value.sema

    @sema.setter
    def sema(self, value):
        self.value.sema = value


    def set_from_string(self, s: str, rounding_mode):
        data = s.encode('ascii')

        pos = 0
        self.sign = 1 if data[pos] == ord('-') else 0
        if data[pos] in [ord('-'), ord('+')]:
            pos += 1

        if len(data) - pos >= 2 and data[0] == ord('0') and (data[1] == ord('x') or data[1] == ord('X')):
            pos += 2
            assert(len(data) - pos > 0)
            return self._set_from_hexadecimal_string(data[pos:], rounding_mode)
            
        return self._set_from_decimal_string(data[pos:], rounding_mode)

    def bitcast_to_apint(self):
        return self.value.bitcast_to_apint()


if __name__ == "__main__":
    a = APInt(128, False)
    a.set_from_string("6565786568769876876363463", 10)

    print(a)

    f = APFloat(FltSemanticsDef.IEEE_Double.value)
    status = f.set_from_string("2.2", FltRoundingMode.TowardZero)

    import struct
    f2 = struct.unpack('d', f.bitcast_to_apint().to_bytes(8, 'little'))
    print(f.bitcast_to_apint())
    print(f)

    a = APInt(32, False)
    a.set_from_int(65363463)

    b = APInt(32, False)
    b.set_from_int(10)

    c = a / b

    num_per_word = 0x1 << APINT_BITS_PER_WORD

    a_val = 0
    for w in reversed(a.values):
        a_val *= num_per_word
        a_val += w

    b_val = 0
    for w in reversed(b.values):
        b_val *= num_per_word
        b_val += w

    c_val = 0
    for w in reversed(c.values):
        c_val *= num_per_word
        c_val += w

    result = a_val / b_val
    print(c)
    

